from mimetypes import init
import torch.nn as nn
import re

class SimpleFeatureSingleModel(nn.Module):
    """
    Apply Layer-Normalization to the input features, before passing them through MLP layers.
    """
    def __init__(self, num_clip_layers, final_linear):
        super(SimpleFeatureSingleModel, self).__init__()
        self.clip_layer_norm = nn.LayerNorm(num_clip_layers)
        self.final_linear    = final_linear
    
    def forward(self, features):
        #Apply LayerNorm to the input features
        v1_sum = self.clip_layer_norm(features)
        #Pass through the final linear layer
        v_hat  = self.final_linear(v1_sum)
        
        return v_hat

#====================================================================================================================================
# Main Multimodal-Projector Builder
#====================================================================================================================================
def build_multimodal_projector(config, **kwargs):
    """
    Build the multimodal projector based on the provided configuration: MLP layers.
    """
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    #MLP Projector
    mlp_gelu_match = re.match(r'mlp(\d+)x_(\w+)', projector_type)
    if mlp_gelu_match:
        layers_depth = int(mlp_gelu_match.group(1))
        mlp_modules  = [
            nn.Linear(config.mm_hidden_size, config.hidden_size)
        ]

        for _ in range(1, layers_depth):
            #GeLU activation function
            mlp_modules.append(nn.GELU())
            #Linear layer
            mlp_modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        
        mlp_modules = nn.Sequential(*mlp_modules)

        return SimpleFeatureSingleModel(
            num_clip_layers = config.mm_hidden_size,
            final_linear    = mlp_modules
        )
    
    raise ValueError(f"Unknown multimodal projector type: {projector_type}")

