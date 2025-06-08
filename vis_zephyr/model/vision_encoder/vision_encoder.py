from numpy import concatenate
from requests import patch
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionTower(nn.Module):
    """CLIP Vision Tower for image processing."""
    def __init__(self, vision_tower_path, args, delay_load = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_path = vision_tower_path
        
        # Feature selection strategy
        self.select_layers  = [-2, -5, -8, -11, 6]
        self.select_feature = 'patch'
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_path)
        
    
    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
        self.vision_tower    = CLIPVisionModel.from_pretrained(self.vision_tower_path)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_output):
        """Select features from the vision tower output."""
        
        #Select features from hidden states
        image_selected_features = [image_forward_output['hidden_states'][i] for i in self.select_layers]
        #Process patch features
        patch_features = [features[:, 1:] for features in image_selected_features]
        #Concatenate patch features along the feature dimension
        concatenated_features = torch.cat(patch_features, dim=-1)

        return concatenated_features

    @torch.no_grad()
    def forward(self, images):
        """Forward pass through the vision tower."""
        if type(images) is list:
            images_features = []
            for image in images:
                image_forward_output = self.vision_tower(image.to(device = self.device, dtype = self.dtype).unsqueeze(0),
                                                         output_hidden_states = True)
                image_features = self.feature_select(image_forward_output).to(image.dtype)
                images_features.append(image_features)
        else:
            image_forward_output = self.vision_tower(images.to(device = self.device, dtype = self.dtype),
                                                     output_hidden_states = True)
            images_features = self.feature_select(image_forward_output).to(images.dtype)
        
        return images_features
    
    @property
    def dtype(self):
        """Return the data type of the vision tower."""
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
        return self.vision_tower.config.hidden_size*len(self.select_layers)
    @property
    def num_patches(self):
        """Return the number of patches in the vision tower."""
        return (self.config.image_size // self.config.patch_size)**2