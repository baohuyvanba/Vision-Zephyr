#==================================================================================================
# File: vis_zephyr/train/train.py
# Description: Main script for training Vision-Zephyr models.
#                - Handle both Stage 1 and Stage 2;
#                - Handles argument parsing, model/tokenizer loading, LoRA setup;
#                - Include data preprocessing, and orchestrates the training process.
#==================================================================================================
import time
import psutil

import os
import copy
from dataclasses import dataclass, field
import json
import pathlib
import random
from typing import Dict, Optional, Sequence

import torch
import transformers
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transformers.trainer_utils import get_last_checkpoint

from vis_zephyr.constants import ( DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN)
from vis_zephyr.model.vip_processor.configuration import visual_prompt_config
from vis_zephyr.model.vip_processor.processor import visual_prompt_process
from vis_zephyr.train.vis_zephyr_trainer import VisZephyrTrainer, maybe_zero
from vis_zephyr import conversation as conv_lb
from vis_zephyr.model import VisZephyrForCausalLM
from vis_zephyr.model.mm_utils import tokenizer_image_token

local_rank = None

def rank0_print(*args):
    """
    Print function that only prints on rank 0 to avoid cluttering the output in distributed training.
    """
    if local_rank == 0:
        print(*args)

def set_seed(seed: int):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

#------------------------------------------------------------------------------------------------------------------------------------
# ARGUMENTS CLASS 
#------------------------------------------------------------------------------------------------------------------------------------
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
    mm_vision_tower: Optional[str] = field(
        default  = "openai/clip-vit-large-patch14-336",
        metadata = {"help": "Vision tower model name or path."}
    )
    mm_vision_select_layer: Optional[str] = field(
        default  = "-2",
        metadata = {"help": "Select layer to get features from multimodal vision tower."}
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default  = None,
        metadata = {"help": "Path to pre-trained multimodal projector (mm_projector.bin)."}
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
    #Anyres
    mm_grid_pinpoints: Optional[str] = field(
        default  = None,
        metadata = {"help": "A string representation of a list of possible resolutions for processing high-resolution images, e.g., '[[336, 672], [672, 336], [336, 1008], [1008, 336]]'."}
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
    lora_enable : bool  = False #turn LoRA on or off
    lora_r      : int   = 64
    lora_alpha  : int   = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias   : str   = "none"

    #Quantization Arguments
    bits: int = field(
        default  = 16,
        metadata = {"help": "Precision: 16, 8, or 4."}
    )

#------------------------------------------------------------------------------------------------------------------------------------
# SAVING & STATE HANDLING
#------------------------------------------------------------------------------------------------------------------------------------
def get_peft_state_maybe_zero(named_parameters, bias):
    """
    Get PEFT state dictionary, handling zero sharding
    """

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

#------------------------------------------------------------------------------------------------------------------------------------
# UTILS FUNCTIONS: LoRA and Model saving
#------------------------------------------------------------------------------------------------------------------------------------
# Find all linear layer names for LoRA targeting, EXCLUDING vision encoder + projector parts.
def find_all_linear_names(model):
    """
    Finds all linear layer names for LoRA targeting, excluding vision encoder + projector parts.
    """
    cls = torch.nn.Linear
    lora_module_names   = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'mm_vision_tower'] #to exclude multimodal components

    for name, module in model.named_modules():
        if any(keyword in name for keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    #Pass lm_head
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

# Saving model weights for multi-stage training
def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer,
    output_dir: str
):
    """
    Handles saving the model, especially for multi-stage training: save states to disk.
    """
    #STAGE 1 - Pretraining stage: save only Projector's weights
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        from vis_zephyr.train.vis_zephyr_trainer import get_mm_adapter_state_maybe_zero
        
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero(trainer.model.named_parameters(), keys_to_match)

        trainer.model.config.save_pretrained(output_dir)
        torch.save(
            weight_to_save,
            os.path.join(output_dir, 'mm_projector.bin')
        )

        # current_folder = output_dir.split('/')[-1]
        # parent_folder  = os.path.dirname(output_dir)
        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        #     if current_folder.startswith('checkpoint-'):
        #         mm_projector_model = os.path.join(parent_folder, "mm_projector")
        #         os.makedirs(mm_projector_model, exist_ok = True)
        #         torch.save(
        #             weight_to_save,
        #             os.path.join(mm_projector_model, f'{current_folder}.bin')
        #         )
        #         rank0_print(f"Saving multimodal projector weights to {os.path.join(mm_projector_model, f'{current_folder}.bin')}")
        #     else:
        #         torch.save(
        #             weight_to_save,
        #             os.path.join(output_dir, f'mm_projector.bin')
        #         )
        return
    
    #STAGE 2 - Finetuning model: both DeepSpeed and non-Deepspeed)
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

#------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------------------
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
            
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_pretrain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conservations = []
    #Create conservation only has: <Image> + Caption
    for source in sources:
        assert len(source) == 2 #Pretrain only has 2 messages: user -> assistant
        assert DEFAULT_IMAGE_TOKEN in source[0]['value'], "Pretrain conversation must start with image token."

        #Ignore the user message (question/prompt)
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conservation = source[0]['value'] + source[1]['value'] + conv_lb.default_conversation.separator_01
        conservations.append(conservation)

    input_ids = [tokenizer_image_token(
        prompt    = prompt,
        tokenizer = tokenizer,
        return_tensors = 'pt'
    ) for prompt in conservations]
    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        #Get the length of the image token
        image_index_token_length = len(tokenizer_image_token(
            prompt    = source[0]['value'],
            tokenizer = tokenizer,
            return_tensors = 'pt'
        ))
        #Mask the image token in the target
        target[:image_index_token_length] = IGNORE_INDEX
    
    return dict(
        input_ids = input_ids,
        labels    = targets,
    )

def preprocess_zephyr(
    sources  : Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Tokenizes conversations and creates labels for supervised learning.
    """
    conv = conv_lb.default_conversation.copy()
    roles_mapping = {
        "human": conv.roles[0],
        "gpt"  : conv.roles[1]
    }

    # --- 1 --- Apply prompt templates
    conversations_list = []
    for i, source in enumerate(sources):
        #If the first message is not from the human ('user') in dataset -> skip.
        if roles_mapping[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        #Append system message if available
        for j, sentence in enumerate(source):
            role = roles_mapping[sentence["from"]]
            assert role == conv.roles[j % 2], f"Conversation role mismatch."
            conv.append_message(role, sentence["value"])
        
        #Get the full conversation prompt
        prompt = conv.get_prompt()
        conversations_list.append(prompt)

    # --- 2 --- Tokenize conversations
    if has_image:
        input_ids = torch.stack([
            tokenizer_image_token(
                prompt         = prompt,
                tokenizer      = tokenizer,
                return_tensors = 'pt'
            )
            for prompt in conversations_list
        ], dim = 0)
    else:
        #Text-only tokenization
        input_ids = tokenizer(
            conversations_list,
            return_tensors = 'pt',
            padding        = "longest",
            max_length     = tokenizer.model_max_length,
            truncation     = True,
        ).input_ids

    #Copy input_ids -> targets (labels)
    targets = input_ids.clone()

    # --- 3 --- Mask targets -> Only calculate Loss only on "Assistant responses"
    system_role_token    = "<|system|>\n"           #<|system|>\n
    user_role_token      = f"<|{conv.roles[0]}|>\n" #<|user|>\n
    assistant_role_token = f"<|{conv.roles[1]}|>\n" #<|assistant|>\n
    assistant_prompt_len = len(tokenizer(assistant_role_token, return_tensors = 'pt').input_ids[0])

    #for conversation, target in zip(conversations_list, targets):
    for idx, (conversation, target) in enumerate(zip(conversations_list, targets)): 
        #Total length of the conversation
        total_length = int(target.ne(tokenizer.pad_token_id).sum())

        #Split conversation into turns (each turn is a Assistant respone/or User prompt)
        turns = conversation.split(conv.separator_01)  #-> ['<|system|>...', '<|user|>...', '<|assistant|>...', '<|user|>...', ...]
        
        current_length          = 1                            #Start from 1 to ignore the first token (BOS token)
        target[:current_length] = IGNORE_INDEX                 #Assign 'IGNORE_INDEX' -> BOS token

        for t_i, turn in enumerate(turns):
            #Skip empty turns
            if turn == "" or not turn:
                break
            
            #Re-addding separator to the turn -> correct tokenized length: '<|user|>...</s>' or '<|assistant|>...</s>'
            turn_with_separator = turn + conv.separator_01

            #MASK (IGNORE_INDEX) system + user -> Only Calculate loss for assistant responses
            not_assistant_turn = system_role_token in turn or user_role_token in turn
            
            if has_image: #and '<image>' in turn_with_separator:
            #Tokenize turn with image
                turn_length = len(tokenizer_image_token(
                    prompt    = turn_with_separator,
                    tokenizer = tokenizer
                )) -2
            else:
            #Tokenize turn without image
                turn_length = len(tokenizer(
                    turn_with_separator,
                    return_tensors = 'pt'
                )['input_ids'][0]) -2 
            
            #Apply IGNORE_INDEX to system and user messages
            if not_assistant_turn:
                target[current_length:current_length + turn_length] = IGNORE_INDEX
            else:
                #Apply IGNORE_INDEX to assistant_tokens <|assistant|>\n
                target[current_length:current_length + assistant_prompt_len] = IGNORE_INDEX

            #Move the cursor to the next turn
            current_length += turn_length

        #Apply IGNORE_INDEX to the rest of the tokens
        target[current_length:] = IGNORE_INDEX

        #If Current != Total length -> Something went wrong -> Mask all tokens for safety
        if current_length < tokenizer.model_max_length:
            if current_length != total_length:
                target[:] = IGNORE_INDEX
                rank0_print(f"WARNING: Tokenization mismatch (cur_len={current_length}, total_len={total_length}). Ignoring sample.")

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
    if conv_lb.default_conversation.separator_style == conv_lb.SeparatorStyle.ZEPHYR:
        return preprocess_zephyr(
            sources   = sources,
            tokenizer = tokenizer,
            has_image = has_image
        )
    elif conv_lb.default_conversation.separator_style == conv_lb.SeparatorStyle.PLAIN:
        return preprocess_pretrain(
            sources   = sources,
            tokenizer = tokenizer
        )
    
    raise ValueError(f"Unsupported conversation version: {conv_lb.default_conversation.version}. Supported: zephyr_v1.")

#------------------------------------------------------------------------------------------------------------------------------------
# DATASET and COLLATOR
#------------------------------------------------------------------------------------------------------------------------------------
class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning (process on the fly).
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer      = tokenizer
        self.data_args      = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]
        
        #All data is multimodal (for Vis-Zephyr)
        if 'image' in sources[0]:
            #Read image
            image_folder  = self.data_args.image_folder
            image_file    = self.list_data_dict[i]['image']
            image         = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            original_size = image.size

            #Vision Encoder's Image processor
            processor = self.data_args.image_processor
            
            #VISUAL PROMPT: Check if there is Visual Prompt in Dataset ---------------------------------------------------------------
            if type(sources[0]['id']) == str and sources[0]['id'].split('-')[0] in visual_prompt_config:            
                try:
                    image, conversations = visual_prompt_process(
                        source            = sources[0],
                        image             = image,
                        image_size_anchor = processor.crop_size['height'],
                        data_args         = self.data_args,
                    )
                except:
                    print(f"=== Error processing ViP ===")
                    return self.__getitem__(random.randint(0, len(self.list_data_dict)-1))
                sources[0]["conversations"] = conversations
            
            #Apply padding to make image square
            if self.data_args.image_aspect_ratio == 'pad':
                from vis_zephyr.model.mm_utils import expand2square
                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                #Preprocess Image using the CLIP processor
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            #Any-res resolution images
            elif self.data_args.image_aspect_ratio == 'anyres':
                from vis_zephyr.model.multi_scale_process import process_any_resolution_image
                grid_pinpoints = getattr(self.data_args, 'mm_grid_pinpoints', None)
                image_tensors_list = process_any_resolution_image(
                    image          = image,
                    processor      = processor,
                    grid_pinpoints = grid_pinpoints
                )
                #Preprocess Patches using the CLIP processor
                # image_tensors_list = [
                #     processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                #     for image in image_tensors_list
                # ]
                image = image_tensors_list
            
            #Image is square
            else:
                #Preprocess Image using the CLIP processor
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            #Preprocess conversations
            conversations = copy.deepcopy([e["conversations"] for e in sources])
            conversations = preprocess_multimodal(conversations, self.data_args) #Replace image tokens -> <image>\n{sentence}
        
        #Text-only conversations
        else:
            conversations = copy.deepcopy([e["conversations"] for e in sources])
        
        #Apply Conversation Template: Tokenize conversations and create labels
        data_dict = preprocess(
            sources   = conversations,
            tokenizer = self.tokenizer,
            has_image = ('image' in self.list_data_dict[i])
        ) #-> dict(input_ids, labels)
        
        if isinstance(i, int):
            data_dict = dict(
                input_ids = data_dict["input_ids"][0],
                labels    = data_dict["labels"][0]
            )
        
        if 'image' in sources[0]:
            data_dict['image'] = image
            #Add original image size for any-resolution images processing
            data_dict['images_size'] = original_size
        else:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width']) # Dummy tensor for text-only samples

        return data_dict

    @property
    def modality_lengths(self):
        """
        Calculate lengths of each multi-modality sample in the dataset.
          - Use Negative values (<0) for text-only samples.
        """
        length_list = []
        for sample in self.list_data_dict:
            #Calculate length based on the number of words in conversations
            current_length = sum(
                len(conv['value'].split())
                for conv in sample['conversations']
            )
            #Value: >0 -> multimodal sample; <0 -> text-only sample
            current_length = -current_length if 'image' not in sample else current_length
            length_list.append(current_length)
        return length_list

    @property
    def lengths(self):
        """
        Calculate lengths of each sample in the dataset.
        """
        length_list = []
        for sample in self.list_data_dict:
            image_token = 128 if 'image' in sample else 0
            length_list.append(sum(
                len(conv['value'].split())
                for conv in sample['conversations']
            ) + image_token)
        
        return length_list

# DATACOLLATOR: to handle padding and batching for supervised fine-tuning
@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning, handling padding and batching.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([
                instance[key]
                for instance in instances
            ] for key in ("input_ids", "labels")
        )
        
        #Pad input_ids to the longest sequence in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first   = True,
            padding_value = self.tokenizer.pad_token_id
        )
        #Pad labels with IGNORE_INDEX 
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first   = True,
            padding_value = IGNORE_INDEX
        )
        
        #Truncate to model max length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels    = labels[:, :self.tokenizer.model_max_length]
        
        #Batch dictionary: input_ids, labels, and attention_mask
        batch = dict(
            input_ids      = input_ids,
            labels         = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
        )

        #If images are present, stack them
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]

            if all(x is not None and x.shape == images[0].shape and isinstance(x, torch.Tensor) for x in images if isinstance(x, torch.Tensor)):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            
            if 'images_size' in instances[0]:
                batch['images_size'] = [instance['images_size'] for instance in instances]

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    """
    train_dataset = LazySupervisedDataset(
        tokenizer = tokenizer,
        data_path = data_args.data_path,
        data_args = data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)

    return dict(
        train_dataset = train_dataset,
        eval_dataset  = None,
        data_collator = data_collator
    )

#------------------------------------------------------------------------------------------------------------------------------------=
# MAIN TRAINING FUNCTION
#------------------------------------------------------------------------------------------------------------------------------------=
def train(
    attn_implementation: str = "flash_attention_2"
):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    set_seed(0)

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if attn_implementation is not None:
        rank0_print(f"Using attention implementation: {attn_implementation}")

    #Pass Arguments to the Data Arguments
    training_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    data_args.mm_use_im_start_end     = model_args.mm_use_im_start_end
    model_args.image_aspect_ratio     = data_args.image_aspect_ratio
    data_args.mm_grid_pinpoints       = model_args.mm_grid_pinpoints
    if model_args.mm_grid_pinpoints is not None:
        rank0_print(f"Using mm_grid_pinpoints: {model_args.mm_grid_pinpoints}")

    # --- 1 --- Set up: Model Loading and Configuration ------------------------------------------------------------------------------
    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32) #Datatype
    )
    
    ###HOOK: Quantization
    
    #Loading model
    model = VisZephyrForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_args.model_name_or_path,
        cache_dir                     = training_args.cache_dir,
        attn_implementation           = attn_implementation,
        torch_dtype                   = compute_dtype,
        #**bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    #Freeze backbone model
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    ###HOOK: Quantization training
    
    # --- 2 --- LoRA Setup (if enabled) ----------------------------------------------------------------------------------------------
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        rank0_print("Enabling LoRA for the model.")
        lora_config = LoraConfig(
            r              = training_args.lora_r,
            lora_alpha     = training_args.lora_alpha,
            target_modules = find_all_linear_names(model),
            lora_dropout   = training_args.lora_dropout,
            bias           = training_args.lora_bias,
            task_type      = "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # --- 3 --- Set up tokenizer -----------------------------------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir        = training_args.cache_dir,
        model_max_length = training_args.model_max_length,
        padding_side     = "right",
        use_fast         = False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Mistral/Zephyr doesn't have a pad token
    
    conv_lb.default_conversation = conv_lb.templates[model_args.version]

    # --- 4 --- Initialize Vision modules --------------------------------------------------------------------------------------------
    model.get_model().initialize_vision_modules(
        model_args = model_args,
    )
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(
        device = training_args.device,
        dtype  = torch.bfloat16 if training_args.bf16 else torch.float16
    )
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal   = True

    # --- 5 --- Configs Multimodal Project Training ----------------------------------------------------------------------------------
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    # > Pretrain: Stage 1
    if model_args.tune_mm_mlp_adapter:
        rank0_print("===== Stage 1: Pre-training Projector =====")
        model.requires_grad_(False)

        #MLP
        if hasattr(model.get_model(), "mm_projector"):
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
                rank0_print(f"Enabling training in mm_projector.")
        else:
            raise ValueError(
                "model.get_model().mm_projector is not available."
            )
    
    # > Fine-tune: Stage 2
    #Freeze MLP adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.require_grad = False

    # --- 6 --- Initialize special visual tokens -------------------------------------------------------------------------------------
    model.initialize_vision_tokenizer(
        model_args = model_args,
        tokenizer  = tokenizer,
    )

    # --- 7 --- Set up Data Module & Trainer -----------------------------------------------------------------------------------------
    data_module = make_supervised_data_module(
        tokenizer = tokenizer,
        data_args = data_args
    )
    trainer = VisZephyrTrainer(
        model     = model,
        tokenizer = tokenizer,
        args      = training_args,
        **data_module
    )

    # --- START BENCHMARK LOGGING ---
    if local_rank in (0, None):
        torch.cuda.reset_peak_memory_stats()
        cpu_mem_before = psutil.virtual_memory().used if psutil else None
        t0 = time.time()
        n_samples = len(trainer.train_dataset)
        n_params_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        rank0_print(f"[BENCH] Trainable params: {n_params_trainable:,}")
    # --- END PREPARE LOGGING ---

    # --- 8 --- TRAINING -------------------------------------------------------------------------------------------------------------
    #Train from Scratch or Resume from Checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    if last_checkpoint is not None and os.path.exists(os.path.join(last_checkpoint, "trainer_state.json")):
        rank0_print(f"Found a valid, resumable checkpoint at {last_checkpoint}. Resuming training.")
        
        if model_args.tune_mm_mlp_adapter:
            projector_weights_path = os.path.join(last_checkpoint, "mm_projector.bin")
            if os.path.exists(projector_weights_path):
                projector_weights = torch.load(projector_weights_path, map_location="cpu")
                model.load_state_dict(projector_weights, strict=False)
                rank0_print(f"  > Manually loaded projector weights from checkpoint.")
        
        trainer.train(resume_from_checkpoint = last_checkpoint)
    else:
        rank0_print("No valid checkpoint found. Starting training from scratch.")
        final_projector_path = os.path.join(training_args.output_dir, "mm_projector.bin")
        if model_args.tune_mm_mlp_adapter and os.path.exists(final_projector_path):
             rank0_print(f"  > Found final weights from a previous completed run. Loading: {final_projector_path}")
             projector_weights = torch.load(final_projector_path, map_location="cpu")
             model.load_state_dict(projector_weights, strict=False)

        trainer.train()

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")): #searching for sub-folder named "checkpoint-*"
    #     rank0_print("Resuming training from the last checkpoint.")
    #     trainer.train(resume_from_checkpoint = True)
    # else:
    #     rank0_print("Starting training from scratch.")
    #     trainer.train()

    # --- END BENCHMARK LOGGING ---
    if local_rank in (0, None):
        t1 = time.time()
        total_time = t1 - t0
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**2)
        cpu_mem_after = psutil.virtual_memory().used if psutil else None
        cpu_delta = cpu_mem_after - cpu_mem_before if cpu_mem_before is not None else None
        throughput = n_samples / total_time
        rank0_print(f"[BENCH] Total time       : {total_time:.1f}s")
        rank0_print(f"[BENCH] Throughput       : {throughput:.1f} samples/s")
        rank0_print(f"[BENCH] GPU peak memory  : {gpu_peak:.1f} MiB")
        if cpu_delta is not None:
            rank0_print(f"[BENCH] CPU delta memory : {cpu_delta/1024**2:.1f} MiB")
    if local_rank in (0, None):
        csv_line = ",".join(map(str, [
            model_args.version,
            n_samples,
            n_params_trainable,
            f"{total_time:.1f} s",
            f"{throughput:.1f} samples/s",
            f"{gpu_peak:.1f} MiB",
            f"{cpu_delta/1024**2:.1f}" if cpu_delta else ""
        ]))
        with open(os.path.join(training_args.output_dir, "benchmark.csv"), "a") as f:
            f.write(csv_line + "\n")
    # --- END LOGGING ---

    trainer.save_state()
    model.config.use_cache = True

    # --- 9 --- Save the FINAL model ------------------------------------------------------------------------------------------------
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero(
            model.named_parameters(),
            training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero(
            model.named_parameters(),
            require_grad_only = True
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir,
                state_dict = state_dict,
            )
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin")
            )
    else:
        #Handle saving the model for Pretraining & Finetuning
        safe_save_model_for_hf_trainer(
            trainer    = trainer,
            output_dir = training_args.output_dir
        )

if __name__ == "__main__":
    train()

# if __name__ == "__main__":
#     train(attn_implementation = "flash_attention_2")
