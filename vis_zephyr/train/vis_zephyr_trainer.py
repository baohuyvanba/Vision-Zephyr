# =================================================================================================
# File: vis_zephyr/train/vis_zephyr_trainer.py
# Description: Trainer for Vis-Zephyr models.
#   - Handles multimodal data sampling, optimizer creation for the Projector;
#   - Checkpoint-saving for 2 stages training.
# =================================================================================================
import os
from sympy import li
import torch
from torch.utils.data import Sampler
from typing import List, Optional

from transformers import Trainer
from transformers.trainer import (
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger
)
from transformers.trainer_pt_utils import get_parameter_names

# =====================================================================================================================================
# UTILITY FUNCTIONS: for DEEPSPEED ZeRO
# =====================================================================================================================================
def maybe_zero(parameters, ignore_status = False, name = None):
    """
    Handle parameters in ZeRO stage:
      - Gathers a parameters from All GPUs;
      - Saving all model's weights in distributed environment.
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    #Parameters are in ZeRO stage
    if hasattr(parameters, "ds_id"):
        if parameters.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
                logger.warning(f"Parameter {name} is not available in ZeRO stage. Skipping.")
        
        #Gather the parameters from all ranks/GPUs
        with zero.GatheredParameters([parameters]):
            parameters = parameters.detach().cpu().clone()
    #Parameters are not in ZeRO stage
    else:
        parameters = parameters.detach().cpu().clone()
    
    return parameters

# Extracts parameters
def get_mm_adapter_state_maybe_zero(named_parameters, keys_to_match):
    """
    Extracts parameters from state dict - match specific keys, handling ZeRO stage parameters.
    """
    to_return = {k: t for k, t in named_parameters if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero(v, ignore_status = True, name = k) for k, v in to_return.items()}
    return to_return

#======================================================================================================================================
# LengthGroupedSampler for Efficiency: groups features of similar lengths.
#======================================================================================================================================
class LengthGroupedSampler(Sampler):
    """
    SAMPLER: Groups features of similar lengths together.
    """
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths   : Optional[List[int]] = None,
        generator = None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided for LengthGroupedSampler.")
        
        self.batch_size = batch_size #Number of samples per batch
        self.world_size = world_size #Number of processes (GPUs) in distributed training
        self.lengths    = lengths    #List of lengths for each sample in the dataset  
        self.generator  = generator  #Random generator for reproducibility
        self.group_by_modality = group_by_modality #Group by modality lengths or not

    def __len__(self):
        return len(self.lengths)
    
    def __iter__(self):
        if self.group_by_modality:
            #Split indices into chunks based on modality lengths
            indices = get_length_grouped_indices_by_modality(
                lengths    = self.lengths,
                batch_size = self.batch_size,
                world_size = self.world_size,
                generator  = self.generator
            )
        else:
            #Simple length grouping
            indices = get_length_grouped_indices(
                lengths    = self.lengths,
                batch_size = self.batch_size,
                world_size = self.world_size,
                generator  = self.generator
            )
        return iter(indices)

# SPLIT EVENLY INDICES -> CHUNKS
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split indices into even chunks based on lengths (divide equally/split evenly)
    """
    if len(indices) % num_chunks != 0:
        return [
            indices[i::num_chunks]
            for i in range(num_chunks)
        ]
    
    number_of_indices_per_chunk = len(indices) // num_chunks
    list_chunks   = [[] for _ in range(num_chunks)]
    chunk_lengths = [0 for _ in range(num_chunks)]

    for indice in indices:
        shortest_chunk = chunk_lengths.index(min(chunk_lengths)) #Find the shortest chunk
        list_chunks[shortest_chunk].append(indice)               #Add the index to the shortest chunk
        chunk_lengths[shortest_chunk] += lengths[indice]         #Update the length of the shortest chunk
        if len(list_chunks[shortest_chunk]) == number_of_indices_per_chunk:
            chunk_lengths[shortest_chunk] = float('inf')         #Set to INF to avoid adding more indices

    return list_chunks

# GENERATE LENGTH GROUPED INDICES
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Generate indices grouped by lengths for efficient batching.
    """
    #Shuffle indices
    indices = torch.randperm(len(lengths), generator = generator).tolist()
    #Number of samples per megabatch = batch_size (per GPU) * world_size (number of GPUs)
    megabatch_size = batch_size * world_size
    #Split indices into megabatches
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    #Sort megabatches by lengths
    megabatches = [sorted(megabatch, key = lambda i: lengths[i], reverse = True) for megabatch in megabatches] 
    #Flatten the megabatches into a single list of indices
    grouped_indices = [idx for megabatch in megabatches for idx in megabatch]
    return grouped_indices

# GENERATE LENGTH GROUPED INDICES BY MODALITY
def get_length_grouped_indices_by_modality(lengths, batch_size, world_size, generator=None):
    """
    Generate indices grouped by modality lengths for efficient batching.
    """
    if all(l != 0 for l in lengths) or all(l < 0 for l in lengths):
        #All samples have the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator)

    #Separate indices based on modality (positive for multimodal, negative for text-only)
    multimodal_indices, multimodal_lengths = zip(*[(idx, length) for idx, length in enumerate(lengths) if length > 0]) #multimodal
    language_indices, language_lengths     = zip(*[(idx, length) for idx, length in enumerate(lengths) if length < 0]) #language-only

    #Shuffle indices
    multimodal_shuffle = [multimodal_indices[i] for i in get_length_grouped_indices(
        lengths    = multimodal_lengths,
        batch_size = batch_size,
        world_size = world_size,
        generator  = generator
    )]
    language_shuffle = [language_indices[i] for i in get_length_grouped_indices(
        lengths    = language_lengths,
        batch_size = batch_size,
        world_size = world_size,
        generator  = generator
    )]

    #Create megabatches for each modality: Internal Length Grouping
    megabatch_size   = batch_size * world_size
    multimodal_megabatches = [multimodal_shuffle[i : i + megabatch_size] for i in range(0, len(multimodal_shuffle), megabatch_size)]
    language_megabatches   = [language_shuffle[i : i + megabatch_size] for i in range(0, len(language_shuffle), megabatch_size)]

    #Handle the last, possible incomplete batch
    last_multimodal  = multimodal_megabatches[-1] if multimodal_megabatches else []
    last_language    = language_megabatches[-1] if language_megabatches else []
    additional_batch = last_multimodal + last_language

    megabatches = (multimodal_megabatches[:-1] if multimodal_megabatches else []) + (language_megabatches[:-1] if language_megabatches else [])

    #Shuffle the megabatches to mix modalities
    megabatch_indices = torch.randperm(len(megabatches), generator = generator)
    megabatches       = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        #Add the last, possibly incomplete batch
        megabatches.append(sorted(additional_batch))

    return [idx for megabatch in megabatches for idx in megabatch]

# =====================================================================================================================================
# VIS-ZEPHYR MAIN TRAINER CLASS
# =====================================================================================================================================
class VisZephyrTrainer(Trainer):
    """
    Trainer for Vis-Zephyr models <- base Trainer class.
      - Handles multimodal inputs (sampling) and custom training logic.
      - Specialized checkpoint saving for multi-stage training.
    """
    def _get_train_sampler(self, train_dataset = None) -> Optional[torch.utils.data.Sampler]:
        """Returns a sampler for the training dataset."""
        if train_dataset is None:
            train_dataset = self.train_dataset

        #Check if the train dataset is available and has length
        if train_dataset is None or not has_length(train_dataset):
            return None
        
        #Use LengthGroupedSampler to group
        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                batch_size = self.args.train_batch_size,                                    #Train batch size
                world_size = self.args.world_size * self.args.gradient_accumulation_steps,  #World size (number of GPUs); Gradient accumulation steps (to avoid too large batches)
                lengths    = lengths,                                                       #List of lengths for each sample in the dataset
                group_by_modality = True,
                generator = torch.Generator(device = 'cpu').manual_seed(self.args.seed)
            )
        else:
            #Default trainer sampler
            return super().__get_trainer_sampler(train_dataset = train_dataset)
        
    def create_optimizer(self):
        """
        Sets up the CUSTOM optimizer for training.
          - Allowing for a custom learning rate for the multimodal projector.
        """
        opt_model = self.model

        if self.optimizer is None:
            #Get the parameters that should have weight decay
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.args.mm_projector_lr is not None:
                projector_parameters = [param for param, _ in opt_model.named_parameters() if 'mm_projector' in param]

                optimizer_grouped_parameters = [
                    {
                        # Parameters that require weight decay and are not part of the projector
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n not in projector_parameters and p.requires_grad
                        ],
                        'weight_decay': self.args.weight_decay,
                    },
                    {
                        # Parameters that do not require weight decay (biases) and not part of the projector
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n not in projector_parameters and p.requires_grad
                        ],
                        'weight_decay': 0.0,
                    },
                    {
                        # Parameters of the multimodal projector with a custom learning rate and require weight decay
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in projector_parameters and p.requires_grad
                        ],
                        'weight_decay': self.args.weight_decay,
                        'lr': self.args.mm_projector_lr,
                    },
                    {
                        # Parameters of the multimodal projector with a custom learning rate and do not require weight decay
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in projector_parameters and p.requires_grad
                        ],
                        'weight_decay': 0.0,
                        'lr': self.args.mm_projector_lr,
                    },
                ]
            # No Custom learning rate for the projector
            else:
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and p.requires_grad
                        ],
                        'weight_decay': self.args.weight_decay,
                    },
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and p.requires_grad
                        ],
                        'weight_decay': 0.0,
                    },
                ]
            
            #Create the optimizer
            optimizer_class, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_class(
                optimizer_grouped_parameters,
                **optimizer_kwargs
            
            )

            return self.optimizer
    
    def _save_checkpoint(self, model, trial, metrics = None):
        """
        Save the model checkpoint.
          - Overridden -> handle multi-stage training checkpoints.
          - In Pretraining Stage: save the mm_projector state only.
        """
        #Default checkpoint saving
        super()._save_checkpoint(model, trial, metrics = metrics)

        #Pretraining Stage: only save the mm_projector state
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            
            checkpoint_dir = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir        = self._get_output_dir(trial = trial)
            output_dir     = os.path.join(run_dir, checkpoint_dir)

            #Only save the mm_projector state (Adapter)
            keys_to_match  = ['mm_projector', 'vision_resampler']
            
            #Add special token keys
            if getattr(self.args, 'mm_use_im_start_end', False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            #Get weights to save
            weigth_to_save = get_mm_adapter_state_maybe_zero(
                self.model.named_parameters(),
                keys_to_match
            )
            
            #Create the output directory
            if self.args.local_rank <= 0:
                # self.model.config.save_pretrained(output_dir)
                torch.save(weigth_to_save, os.path.join(output_dir, 'mm_projector.bin'))
                logger.info(f"Saved mm_projector state to {output_dir}/mm_projector.bin")
        
        # #Default checkpoint saving
        # else:
        #     super(VisZephyrTrainer)._save_checkpoint(model, trial, metrics = metrics)

    def _save(
        self,
        output_dir: Optional[str],
        state_dict = None
    ):
        """
        Handle the final model saving - at the end of training.
        """
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super()._save(output_dir = output_dir, state_dict = state_dict)
