# =================================================================================================
# File: vis_zephyr/model/vis_zephyr_arch.py
# Description: CORE Architecture file - Defines the meta-architecture for integrating vision and language models.
# =================================================================================================
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vis_zephyr.model.multi_scale_process import calculate_grid_shape, unpad_image

from .vision_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_multimodal_projector
from ..constants import (IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

from vis_zephyr.model import *

#------------------------------------------------------------------------------------------------------------------------------------
# Abstract class for the core model structure, handling vision components - Be mixed into main LLM Model class
#------------------------------------------------------------------------------------------------------------------------------------
class VisZephyrMetaModel:
    """
    Class for the core model structure, handling vision components (vision encoder, multimodal projector).
    """
    def __init__(self, config):
        """Initialize the VisZephyrMetaModel with the given configuration."""
        super(VisZephyrMetaModel, self).__init__(config)
        
        #Check model's Config for vision tower and multimodal projector 
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_multimodal_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                # If using unpad patch merge type, initialize the image_newline parameter
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype = self.dtype)
                )

    def get_vision_tower(self):
        """Returns the vision tower instance of the model."""
        vision_tower = getattr(self, 'vision_tower', None)
    
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        Initializes the vision modules (vision tower and multimodal projector) based on the model arguments.
        Will be called after the main language model class is initialized.
        """
        vision_tower             = model_args.mm_vision_tower
        mm_vision_select_layer   = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        mm_projector_train       = model_args.pretrain_mm_mlp_adapter #"Checkpoints/vis-zephyr-7b-v1-pretrain/checkpoint-1/mm_projector.bin"
        mm_patch_merge_type      = model_args.mm_patch_merge_type

        #Set config attributes for vision tower
        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = [vision_tower] if fsdp and len(fsdp) > 0 else vision_tower
        else:
            vision_tower = self.vision_tower[0] if fsdp and len(fsdp) > 0 else self.vision_tower
            vision_tower.load_model()

        #Model Configuration attributes
        self.config.use_mm_proj              = True
        self.config.mm_projector_type        = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size           = self.vision_tower.hidden_size
        self.config.mm_vision_select_layer   = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type      = mm_patch_merge_type
        self.config.mm_grid_pinpoints        = getattr(model_args, 'mm_grid_pinpoints', None)
        self.config.image_aspect_ratio       = getattr(model_args, 'image_aspect_ratio', 'square')
        self.config.mm_use_im_start_end      = getattr(model_args, 'mm_use_im_start_end', False)

        #Initialize the Multimodal Projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_multimodal_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype = self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype = self.dtype) * embed_std
                )
        else:
            #Unfreeze the Projector if it was frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
        
        #Load pre-trained Projector weights if available
        if mm_projector_train is not None:
            projector_weights = torch.load(mm_projector_train, map_location='cpu')
            
            def get_w(weights, keyword):
                """Extract weights for a specific keyword."""
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(projector_weights, 'mm_projector'))

# -----------------------------------------------------------------------------------------------------------------------------------
# Abstract class for the full Causal LM, handling the integration of vision and language modalities.
# -----------------------------------------------------------------------------------------------------------------------------------
class VisZephyrMetaForCausalLM(ABC):
    """
    An abstract class for the full Causal Language Modeling, handling the integration of vision and language modalities.
    """
    @abstractmethod
    def get_model(self):
        """Abstract method to get the model instance."""
        pass

    def get_vision_tower(self):
        """Returns the vision tower instance of the model."""
        return self.get_model().get_vision_tower()

    def encode_images(self, images, text_embeddings):
        """Encodes images using the Vision Encoder & Multimodal Projector of the model."""
        image_features     = self.get_model().get_vision_tower()(images)
        projected_features = self.get_model().mm_projector(image_features, text_embeddings=text_embeddings)
        return projected_features

    #----------------------------------------------------------------------------------------------------------------------------------
    # Prepare Inputs and Labels for Multimodal
    #----------------------------------------------------------------------------------------------------------------------------------
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        images_size = None,
    ):
        """
        Prepares inputs and labels for multimodal training by:
          - Encoding Images (Patches) -> Features
          - Projecting Features -> Projected Features
          - Combining Text and Image Features -> New Input Embeddings
        """
        vision_tower = self.get_vision_tower()
        
        #Pure LLM without images
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        #Get text embeddings only
        text_embeddings = []
        for cur_input_ids in input_ids:
            text_only_ids = cur_input_ids[cur_input_ids != IMAGE_TOKEN_INDEX]
            
            cur_text_embed = self.get_model().embed_tokens(text_only_ids)
            
            if self.device is not None:
                cur_text_embed = cur_text_embed.to(self.device)
            
            text_embeddings.append(cur_text_embed)
        
        # --- 1 --- EMBEDDING IMAGE: Image -> Features -> Projected Features ----------------------------------------------------------------------
        if type(images) is list or images.ndim == 5:
            #If images is a list of tensors or a 5D tensor -> process them separately
            # - images can be: list of images (Batch, C, H, W)
            # - 5D Tensor: (Batch, Num_Patches, C, H, W)
            
            if type(images) is list:
                #If images is a list, ensure all images are tensors is (1, C, H, W) or (Num_Patches, C, H_patch, W_patch) in the list
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            #Images list: [B, 1 or Num_Patches, C, H, W]] -> Cat: [B * Num_Patches, C, H, W]
            concat_images  = torch.cat([image for image in images], dim=0)
            
            #IMAGE ENCODING & PROJECTING
            image_features = self.encode_images(concat_images, text_embeddings)

            split_sizes    = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            #image_features: [B * Num_Patches, feature_dim] including feature from image, base image and its sub-patches

            #PATCHES PROCESSING with Image Features
            image_features = self._process_image_patches(
                batched_image_features = image_features,
                images_size            = images_size
            )
        else:
            #If images is a single tensor (C, H, W) - only 1 image
            image_features = self.encode_images(images, text_embeddings)

        # --- 2 --- COMBINE TEXT + IMAGE EMBEDDINGS ---------------------------------------------------------------------------------------------
        #Dummy Tensors
        _labels         = labels
        _position_ids   = position_ids
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype = torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        if position_ids is None:
            position_ids   = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels         = torch.full_like(input_ids, IGNORE_INDEX)

        #Replacing IMAGE_TOKEN_INDEX = Image Features (input_ids)
        new_input_embeds = [] #batch of new input embeddings
        new_labels       = [] #batch of new labels       
        cur_image_indice = 0  #Current index for image features

        #Remove padding if attention_mask is provided
        input_ids  = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
            #Input ids are now is list of tensors with shape (batch_size, seq_length)
        labels     = [curent_labels[cur_attention_mask] for curent_labels, cur_attention_mask in zip(labels, attention_mask)]

        for batch_idx, cur_input_ids in enumerate(input_ids):
            #Check if there is an 'IMAGE_TOKEN_INDEX' (placeholder) in the current input_ids
            num_image_token = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_image_token == 0:
                #No 'IMAGE_TOKEN_INDEX' -> Text-only sample within multimodal batch
                cur_image_features = image_features[cur_image_indice]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds_2 = None
                cur_input_embeds = torch.cat([
                    cur_input_embeds_1,
                    cur_image_features[0:0],
                    ], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_indice += 1
                continue
                
            #Replace 'IMAGE_TOKEN_INDEX' -> image features
            image_token_indices  = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # -1 is added to the beginning to handle the case where the first token is an image token
            
            cur_input_ids_noimage = []
            cur_labels_noimage    = []
            cur_labels            = labels[batch_idx]
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noimage.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noimage.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            #Save the input_ids chunks size without image tokens -> to add image features later
            split_sizes = [x.shape[0] for x in cur_input_ids_noimage]

            #Embeding the text chunks without image tokens
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noimage))
            cur_input_embeds_noimage = torch.split(cur_input_embeds, split_sizes, dim = 0)

            cur_input_embeds_with_image = [] #New input embeddings with image features
            cur_labels_with_image       = [] #New labels with image features

            #Insert Image embeddings into the input embeddings
            for i in range(num_image_token + 1):
                #Add text chunk embeddings/labels
                cur_input_embeds_with_image.append(cur_input_embeds_noimage[i])
                cur_labels_with_image.append(cur_labels_noimage[i])

                if i < num_image_token:
                    cur_image_features = image_features[cur_image_indice]
                    cur_input_embeds_with_image.append(cur_image_features)
                    cur_labels_with_image.append(torch.full(
                        (cur_image_features.shape[0], ),
                        IGNORE_INDEX,
                        dtype  = cur_labels.dtype,
                        device = cur_labels.device
                    ))
                    cur_image_indice  += 1
            
            #Concatenate all input embeddings and labels with image features
            cur_input_embeds_with_image = [x.to(self.device) for x in cur_input_embeds_with_image]
            cur_input_embeds_with_image = torch.cat(cur_input_embeds_with_image)

            cur_labels_with_image = torch.cat(cur_labels_with_image)

            new_input_embeds.append(cur_input_embeds_with_image)
            new_labels.append(cur_labels_with_image)

        #Truncate/Padding the batch to max length and Collate
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        
        #Truncate
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels       = [x[:tokenizer_model_max_length] for x in new_labels]

        new_input_embeds_padded, new_labels_padded, attention_mask, position_ids = self._pad_and_collate_multimodal_inputs(
            new_input_embeds = new_input_embeds,
            new_labels       = new_labels,
            attention_mask   = attention_mask,
            position_ids     = position_ids,
        )

        #Finalize
        if _attention_mask is not None:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        return (
            None,
            position_ids if _position_ids is not None else None,
            attention_mask if _attention_mask is not None else None,
            past_key_values,
            new_input_embeds_padded,
            new_labels_padded if _labels is not None else None,
        )

    #----------------------------------------------------------------------------------------------------------------------------------
    # Initialize Vision Tokenizer
    #----------------------------------------------------------------------------------------------------------------------------------
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
        Add image special placeholder tokens to the tokenizer based on model arguments.
        """
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN],
                special_tokens = True
            )
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens = True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings  = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg  = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:]  = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_intput_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            
            if model_args.pretrain_mm_mlp_adapter:
                #Load pre-trained adapter weights if available
                projector_weights   = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. New tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            #If only using patch tokens, freeze both input and output embeddings if tuning the adapter
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
    
    #----------------------------------------------------------------------------------------------------------------------------------
    # Utility Functions
    #----------------------------------------------------------------------------------------------------------------------------------

    #Process Patches of High-Resolution Images
    def _process_image_patches(
        self,
        batched_image_features,
        images_size,
    ):
        """
        Process Batch of patches image features
        """
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')  #flat/spatial/unpad_spatial
        image_aspect_ratio  = getattr(self.config, 'image_aspect_ratio', 'square') #square/anyres

        if mm_patch_merge_type == 'flat':
            #Flatten the image features for each patch in the batch.
            batched_image_features = [
                feature.flatten(0, 1)
                for feature in batched_image_features
            ]
            return batched_image_features
        elif mm_patch_merge_type.startswith('spatial'):
            #Process the image features as spatial patches.
            new_features = []
            for indice, feature in enumerate(batched_image_features):
                #The image has patches -> Base image + Sub Patches
                if feature.shape[0] > 1:
                    base_feature    = feature[0]  #Base image feature
                    patches_feature = feature[1:] #Sub patches features
                    
                    h = w = self.get_vision_tower().num_patches_per_side
                    assert h*w == base_feature.shape[0]

                    if image_aspect_ratio == 'anyres':
                        num_patch_width, num_patch_height = calculate_grid_shape(
                            image_size = images_size[indice],
                            grid_pinpoints = self.config.mm_grid_pinpoints,
                            patch_size = self.get_vision_tower().config.image_size,
                        )
                        patches_feature = patches_feature.view(num_patch_height, num_patch_width, h, w, -1)
                    else:
                        raise NotImplementedError

                    if 'unpad' in mm_patch_merge_type:
                        patches_feature = patches_feature.permute(4, 0, 2, 1, 3).contiguous()
                        patches_feature = patches_feature.flatten(1, 2).flatten(2, 3)
                        #Unpad the patches feature
                        patches_feature = unpad_image(
                            image_tensor  = patches_feature,
                            original_size = images_size[indice]
                        )
                        patches_feature = torch.cat((
                            patches_feature,
                            self.model.image_newline[:, None, None].expand(*patches_feature.shape[:-1], 1).to(patches_feature.device)
                            ), dim = -1
                        )
                        patches_feature = patches_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        patches_feature = patches_feature.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 3)
                    
                    feature = torch.cat((
                        base_feature,
                        patches_feature,
                        ), dim = 0
                    )
                
                #Single image without patches
                else:
                    feature = feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        feature = torch.cat((
                            feature,
                            self.model.image_newline[None].to(feature.device)
                        ), dim=0)

                new_features.append(feature)

            batched_image_features = new_features
            return batched_image_features
        else:
            raise ValueError(f"Unknown mm_patch_merge_type: {mm_patch_merge_type}")
    
    #Pad and Collate Multimodal Inputs
    def _pad_and_collate_multimodal_inputs(
        self,
        new_input_embeds,
        new_labels,
        attention_mask,
        position_ids,
    ):
        """
        Pads and collates the multimodal inputs, labels, attention mask, and position ids.
        """
        #Maximum length of the input embeddings (after adding image features)
        max_length = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        #Padded Tensors
        padded_input_embeds = torch.zeros(
            (batch_size, max_length, self.config.hidden_size),
            dtype  = new_input_embeds[0].dtype,
            device = new_input_embeds[0].device
        )
        padded_labels = torch.full(
            (batch_size, max_length), IGNORE_INDEX,
            dtype  = new_labels[0].dtype,
            device = new_labels[0].device
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype  = attention_mask.dtype,
            device = attention_mask.device
        )
        padded_position_ids = torch.zeros(
            (batch_size, max_length),
            dtype  = position_ids.dtype,
            device = position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_length = cur_new_embed.shape[0]

            if getattr(self.config, 'tokenizer_padding_side', 'right') == 'left':
                #Left padding (sometimes required for the model generation)
                padded_input_embeds[i, -cur_length:] = cur_new_embed
                if cur_length > 0:
                    padded_labels[i, -cur_length:]         = cur_new_labels
                    padded_attention_mask[i, -cur_length:] = True
                    padded_position_ids[i, -cur_length:]   = torch.arange(0, cur_length, dtype = position_ids.dtype, device = position_ids.device)
            else:
                #Right padding (default)
                padded_input_embeds[i, :cur_length] = cur_new_embed
                if cur_length > 0:
                    padded_labels[i, :cur_length]         = cur_new_labels
                    padded_attention_mask[i, :cur_length] = True
                    padded_position_ids[i, :cur_length]   = torch.arange(0, cur_length, dtype = position_ids.dtype, device = position_ids.device)
        
        return padded_input_embeds, padded_labels, padded_attention_mask, padded_position_ids