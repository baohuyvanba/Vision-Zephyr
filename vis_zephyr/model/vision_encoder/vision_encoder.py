# =================================================================================================
# File: vis_zephyr/model/vision_encoder/vision_encoder.py
# Description: Implements the CLIP-based vision tower for multilayer feature extraction.
# =================================================================================================
from numpy import concatenate
from requests import patch
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from vis_zephyr.model.gating_fusion import DenseChannelIntegrationFusion

class CLIPVisionTower(nn.Module):
    """
    CLIP Vision Encoder Multilayer-Tower for Image Embedding.
    """
    def __init__(self, vision_tower_path, args, delay_load = False):
        super().__init__()
        #
        self.is_loaded = False
        self.vision_tower_path = 'openai/clip-vit-large-patch14-336' if vision_tower_path is None else vision_tower_path
        
        #Feature selection strategy
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        #Feature layers selection
        raw_select_layers = args.mm_vision_select_layer
        if isinstance(raw_select_layers, str):
            try:
                self.select_layers = [int(x.strip()) for x in raw_select_layers.split(',')]
            except ValueError:
                raise ValueError(f"Invalid format for mm_vision_select_layer. Expected a comma-separated string of integers, but got: {raw_select_layers}")
        else:
            self.select_layers = [-2]
        
        #Gating MLP Fusion
        self.gating_fusion = None #load in load_model()

        if not delay_load:
            self.load_model()
        else:
            # If delay_load is True, only load the configuration
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_path)        
    
    def load_model(self):
        """Load the Vision Encoder + Image Processor + Gating MLP Fusion."""
        #Load Image Processor and Vision Tower
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
        self.vision_tower    = CLIPVisionModel.from_pretrained(self.vision_tower_path)

        self.gating_fusion = DenseChannelIntegrationFusion(
            num_groups = 4
        )

        #Freeze the vision tower parameters
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_output):
        """Select features from the vision tower output."""
        hidden_states = image_forward_output['hidden_states']
        
        #Get Selected features
        # 5 layers: selected_features = [image_forward_output['hidden_states'][indice] for indice in self.select_layers]
        selected_features = hidden_states[-(4*5+1):]
        
        if self.select_feature == 'patch': #Default
            #Process Patch features: remove the first feature (CLS token, represent for the whole image)
            patch_features    = [features[:, 1:] for features in selected_features]
        elif self.select_feature == 'cls_patch':
            #Concatenate patch features along the feature dimension
            patch_features    = selected_features
        else:
            raise ValueError(f"Unknown feature selection strategy: {self.select_feature}")

        #Gating Fusion
        fused_features = self.gating_fusion(patch_features)

        return fused_features

    @torch.no_grad()
    def forward(self, images):
        """
        Forward pass through the vision tower.
        """
        if isinstance(images, list):
            images_features_list = []
            
            for image in images:
                #Embedding image
                image_forward_output = self.vision_tower(image.to(
                    device = self.device,
                    dtype = self.dtype
                    ),
                    output_hidden_states = True
                )
                #Select features -> features list
                image_features = self.feature_select(image_forward_output).to(image.dtype)
                images_features_list.append(image_features)
        elif images.ndim == 4:
            #Batched images with [batch_size, channels, height, width]
            image_forward_output = self.vision_tower(images.to(
                device = self.device,
                dtype  = self.dtype),
                output_hidden_states = True
            )
            images_features_list = self.feature_select(image_forward_output).to(images.dtype)
        else:
            #Single image with [channels, height, width]
            images = images.unsqueeze(0)
            image_forward_output = self.vision_tower(images.to(
                device = self.device,
                dtype  = self.dtype),
                output_hidden_states = True
            )
            images_features_list = self.feature_select(image_forward_output).to(images.dtype)
        
        return images_features_list
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device = self.device, dtype = self.dtype)

    @property
    def dtype(self):
        """Return the data type of the vision tower."""
        # if self.gating_fusion is not None:
        #     return self.gating_fusion.dtype
        return self.vision_tower.dtype
    
    @property
    def device(self):
        """Return the device of the vision tower."""
        return self.vision_tower.device
    
    @property
    def config(self):
        """Return the configuration of the vision tower."""
        if not self.is_loaded:
            return self.cfg_only
        else:
            return self.vision_tower.config
    
    @property
    def hidden_size(self):
        """Return the hidden size of the vision tower."""
        return self.vision_tower.config.hidden_size * 5 #len(self.select_layers)
    
    @property
    def num_patches(self):
        """Return the number of patches in the vision tower."""
        return (self.config.image_size // self.config.patch_size)**2