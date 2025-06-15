# =================================================================================================
# File: vis_zephyr/train/train.py
# Description: Main script for training Vision-Zephyr models.
#                - Handle both Stage 1 and Stage 2;
#                - Handles argument parsing, model/tokenizer loading, LoRA setup;
#                - Include data preprocessing, and orchestrates the training process.
# =================================================================================================
from calendar import c
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from re import A
from typing import Dict, Optional, Sequence, List

from attr import has
import torch
import transformers
from torch.utils.data import Dataset

from vis_zephyr.constants import *
from vis_zephyr.train.vis_zephyr_trainer import VisZephyrTrainer
from vis_zephyr import conversation as conv_lb
from vis_zephyr.model import VisZephyrForCausalLM
from vis_zephyr.model.mm_utils import tokenizer_image_token

from PIL import Image

local_rank = None

def rank0_print(*args):
    """
    Print function that only prints on rank 0 to avoid cluttering the output in distributed training.
    """
    if local_rank == 0:
        print(*args)

# ===================================================================================================================================
# ARGUMENTS CLASS 
# ===================================================================================================================================
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default = "HuggingFaceH4/zephyr-7b-beta",
        metadata = {"help": "Path to the pre-trained model or model identifier from HuggingFace."}
    )
    version: Optional[str] = field(
        default  = "zephyr_v1",
        metadata = {"help": "Version of the model, used for conversation template."}
    )
    freeze_backbone: bool = field(
        default  = False,
        metadata = {"help": "Freeze the backbone model during training."}
    )
    tune_mm_mlp_adapter: bool = field(
        default  = False,
        metadata = {"help": "Tune the multimodal MLP adapter, True in pretraining stage."}
    )
    vision_tower: Optional[str] = field(
        default  = "openai/clip-vit-large-patch14-336",
        metadata = {"help": "Vision tower model name or path."}
    )
    mm_vision_select_layer: Optional[int] = field(
        default  = -2,
        metadata = {"help": "Select layer to get features from multimodal vision tower."}
    )
    pretrain_mm_projector: Optional[str] = field(
        default  = None,
        metadata = {"help": "Path to pre-trained multimodal projector."}
    )
    mm_projector_type: Optional[str] = field(
        default  = "mlp2xgelu",
        metadata = {"help": "Type of multimodal projector."}
    )
    mm_use_im_start_end: bool = field(
        default  = False,
        metadata = {"help": "Use image start and end tokens in multimodal training."}
    )
    mm_use_im_patch_token: bool = field(
        default  = True,
        metadata = {"help": "Use image patch token in multimodal training."}
    )
    mm_vision_select_feature: Optional[str] = field(
        default  = "patch",
        metadata = {"help": ""}
    )
    mm_patch_merge_type: Optional[str] = field(
        default  = "flat",
        metadata = {"help": "Type of patch merging for multimodal vision tower."}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default  = None,
        metadata = {"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(
        default = None,
        metadata = {"help": "Folder containing images for multimodal training."}
    )
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    #Training Arguments
    cache_dir: Optional[str] = field(
        default  = None,
        metadata = {"help": "Where to store the pre-trained models downloaded from huggingface.co"}
    )
    optim: str = field(
        default  = "adamw_torch",
        metadata = {"help": "Optimizer to use during training."}
    )
    remove_unused_columns: bool = field(
        default  = False,
        metadata = {"help": "Whether to remove unused columns in the dataset."}
    )
    freeze_mm_mlp_adapter: bool = field(
        default  = False,
        metadata = {"help": "Freeze the multimodal MLP adapter during training."}
    )
    model_max_length: int = field(
        default  = 2048,
        metadata = {"help": "Maximum sequence length for the model."}
    )
    
    bits: int = field(default=16, metadata={"help": "Precision: 16, 8, or 4."})
    
    #Multimodal Arguments
    mm_projector_lr: Optional[float] = field(
        default  = None,
        metadata = {"help": "Learning rate for the multimodal projector."}
    )
    group_by_modality_length: bool = field(
        default  = False,
        metadata = {"help": "Group samples by modality length during training."}
    )

    #LoRA Arguments
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

# ===================================================================================================================================
# SAVING & STATE HANDLING
# ===================================================================================================================================
def get_peft_state_maybe_zero(named_parameters, bias):
    """
    Get PEFT state dictionary, handling zero sharding
    """
    from vis_zephyr.train.vis_zephyr_trainer import maybe_zero

    if bias == "none":
        #Get all LoRA parameters
        to_return = {k: t for k, t in named_parameters if "lora_" in k}
    elif bias == "all":
        #Get all LoRA parameters and biases
        to_return = {k: t for k, t in named_parameters if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return       = {}
        maybe_lora_bias = {}
        lora_bias_name  = set()
        
        for k, t in named_parameters:
            if "lora_" in k:
                to_return[k] = t
                lora_bias_name.add(k.split("lora_")[0] + "bias")
            elif "bias" in k:
                maybe_lora_bias[k] = t
        
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_name:
                to_return[k] = t
    else:
        raise NotImplementedError
    
    return {k: maybe_zero(v) for k, v in to_return.items()}

def get_peft_state_non_lora_maybe_zero(named_parameters, require_grad_only = True):
    """
    Get non-LoRA parameters, handling zero sharding.
    """
    to_return = {k: t for k, t in named_parameters if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    
    return {k: maybe_zero(v) for k, v in to_return.items()}    

# ===================================================================================================================================
# UTILS FUNCTIONS
# ===================================================================================================================================
def find_all_linear_names(model):
    """
    Finds all linear layer names for LoRA targeting, excluding vision encoder + projector parts.
    """
    cls = torch.nn.Linear
    lora_module_names   = set()
    multimodal_keywords = ['mm_projector', 'vision_tower'] #to exclude multimodal components

    for name, module in model.named_modules():
        if any(keyword in name for keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer,
    output_dir: str
):
    """
    Handles saving the model, especially for multi-stage training: save states to disk.
    """

    #Pretraining stage: save only Projector's weights
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        from vis_zephyr.train.vis_zephyr_trainer import get_mm_adapter_state_maybe_zero
        
        keys_to_match = ['mm_projector']
        weight_to_save = get_mm_adapter_state_maybe_zero(trainer.model.named_parameters(), keys_to_match)
        # if getattr(trainer.args, "use_im_start_end", False):
        #     keys_to_match.extend(['embed_tokens', 'embed_in'])

        if trainer.args.local_rank <= 0:
            torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
            rank0_print(f"Saved multimodal projector weights to {os.path.join(output_dir, 'mm_projector.bin')}")
        return
    
    #Traning stage 2 (both DeepSpeed and non-Deepspeed)
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    
    #Standard saving for HF Trainer
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            k: v.cpu()
            for k, v in state_dict.items()
        }
        del state_dict  # Free memory
        trainer._save(output_dir, state_dict = cpu_state_dict)

# ===================================================================================================================================
# PREPROCESSING FUNCTIONS
# ===================================================================================================================================
def preprocess_multimodal(
    sources  : Sequence[str],
    data_args: DataArguments
) -> Sequence[str]:
    """
    Prepares multimodal conversations by adding image tokens.
    """
    if not data_args.is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value'].strip()
    return sources

def preprocess_zephyr(
    sources  : Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Tokenizes conversations and creates labels for supervised learning.
    """
    conv  = conv_lb.default_conversation.copy()
    roles = {
        "human": conv.roles[0],
        "gpt"  : conv.roles[1]
    }

    #Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        #If the first message is not from the human ('user') -> skip.
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        #Append system message if available
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"Conversation role mismatch at turn {j} for sample {i}."
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    #Tokenize conversations
    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors = 'pt')
            for prompt in conversations
        ]
    else:
        #Text-only tokenization
        input_ids = tokenizer(
            conversations,
            return_tensors = 'pt',
            padding        = "longest",
            max_length     = tokenizer.model_max_length,
            truncation     = True,
        ).input_ids

    targets = copy.deepcopy(input_ids)

    #Mask targets -> Only calculate Loss for Assistant responses
    assistant_role_token = f"<|{conv.roles[1]}|>\n"

    for conservation, target in zip(conversations, targets):
        total_length = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conservation.split(conv.separator_02)
        current_length = 0
        target[:1] = IGNORE_INDEX  #Ignore the first token (BOS token)

        for i, turn in enumerate(turns):
            #Skip empty turns
            if turn == "":
                break
            turn_with_separator = turn + conv.separator_02

            #ASSISTANT response
            parts = turn_with_separator.split(assistant_role_token)
            if len(parts) != 2:
                turn_length = len(tokenizer(turn_with_separator, tokenizer))
                targets[current_length + 1:current_length + turn_length] = IGNORE_INDEX
                current_length += turn_length
                continue
            
            #HUMAN response
            user_part, assistant_part = parts
            user_part += assistant_role_token

            #Fisrt turn must have Image
            if has_image and i == 0:
                instruction_length = len(tokenizer_image_token(user_part, tokenizer))
                turn_length        = len(tokenizer_image_token(part_with_separator, tokenizer))
            else:
                instruction_length = len(tokenizer(user_part, return_tensors = 'pt').input_ids[0])
                turn_length        = len(tokenizer(turn_with_separator, return_tensors = 'pt').input_ids[0])

            target[current_length:current_length + instruction_length] = IGNORE_INDEX
            current_length += instruction_length

        target[current_length:] = IGNORE_INDEX

        if current_length < tokenizer.model_max_length:
            if current_length != total_length:
                target[:] = IGNORE_INDEX  #If the conversation is too short, mask all tokens
                rank0_print(f"WARNING: Tokenization mismatch (cur_len={cur_len}, total_len={total_len}). Ignoring sample.")

    return dict(
        input_ids = input_ids,
        labels    = targets,
    )

def preprocess(
    sources   : Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Main Preprocessing function that handles both text and multimodal data.
    """
    if conv_lb.default_conversation.version == "zephyr_v1":
        return preprocess_zephyr(sources, tokenizer, has_image=has_image)
    
    raise ValueError(f"Unsupported conversation version: {conv_lb.default_conversation.version}. Supported: zephyr_v1.")

# ===================================================================================================================================
# DATASET and COLLATOR
# ===================================================================================================================================
class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning (process on the fly).
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments
    ):
        super(LazySupervisedDataset, self).__init__()
        
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]
        
        #All data is assumed to be multimodal (for Vis-Zephyr)
        image_file   = self.list_data_dict[i]['image']
        image_folder = self.data_args.image_folder
        processor    = self.data_args.image_processor
        #Open the image
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        
        #Apply padding to make image square
        if self.data_args.image_aspect_ratio == 'pad':
            from vis_zephyr.model.mm_utils import expand2square
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        
        #Preprocess the image using the CLIP processor
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        #Preprocess conversations
        conversations = copy.deepcopy([e["conversations"] for e in sources])
        conversations = preprocess_multimodal(conversations, self.data_args)
        
        #Tokenize conversations and create labels
        data_dict = preprocess_zephyr(conversations, self.tokenizer)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict['image'] = image
        
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    """
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# ====================================================================================================================================
# MAIN TRAINING FUNCTION
# ====================================================================================================================================
def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set up model
    model = VisZephyrForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir           = training_args.cache_dir,
        #attn_implementation = "flash_attention_2",
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # Set up tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Mistral/Zephyr doesn't have a pad token
    
    conv_lb.default_conversation = conv_lb.conv_templates[model_args.version]

    # Initialize vision modules
    model.get_model().initialize_vision_modules(model_args=model_args)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=training_args.device, dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
    data_args.image_processor = vision_tower.image_processor

    # Configure model for multimodal training
    model.config.mm_vision_select_feature = data_args.is_multimodal = True
    
    # Logic for multi-stage training
    if model_args.tune_mm_mlp_adapter:
        rank0_print("Stage 1: Pre-training multimodal projector.")
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    
    if model_args.lora_enable:
        rank0_print("Stage 2: Fine-tuning with LoRA.")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    # Load data and start training
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = VisZephyrTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()