# =================================================================================================
# File: vis_zephyr/model/vision_encoder/gatingmlp.py
# Description: Implements a Gating MLP mechanism for fusing features from multiple layers.
# =================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseGatedFusion(nn.Module):
    """
    Applies channel-wise feature reweighting using: 
      - Squeeze-and-Excitation (SE);
      - GLU-inspired gating mechanism.
    """
    def __init__(self, channel_dim, reduction=16):
        super().__init__()
        reduced_dim = channel_dim // reduction

        #Squeeze-and-Excitation (SE) layers (MLP layers)
        self.fc1 = nn.Linear(channel_dim, reduced_dim, bias = False)
        self.fc2 = nn.Linear(reduced_dim, channel_dim, bias = False)
        
        #Gated Linear Unit (GLU) layer: channel's importance
        self.gate = nn.Linear(channel_dim, channel_dim, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: [B, P, D] - B: batch, P: patches, D: dim
        B, P, C = x.shape
        
        #1. Global average pooling across the patch dimension (P) -> Global context: [B, D]
        pooled_patch = x.mean(dim = 1)
        
        #2. SE path
        se = F.relu(self.fc1(pooled_patch))  # -> [D, D//r]
        se = torch.sigmoid(self.fc2(se))     # -> [D//r, D]
        
        #3. Gating path 
        gate_weights = torch.sigmoid(self.gate(pooled_patch))  # -> [B, D]
        
        #4. Combine weights and apply to the input tensor
        combined_weight = (se * gate_weights).unsqueeze(1)     # -> [B, 1, D]

        return x * combined_weight

class MultiLayerFeatureFusion(nn.Module):
    """
    Fuses multiple feature tensors (from different layers) using channel-wise gated fusion
    -> Concatenates the results along the channel dimension.
    """
    def __init__(self, num_layers: int, channel_dim: int, reduction: int = 16):
        super().__init__()
        self.num_layers  = num_layers
        self.channel_dim = channel_dim
        
        # ChannelWiseGatedFusion module ~ each input feature
        self.gated_modules = nn.ModuleList([
            ChannelWiseGatedFusion(channel_dim, reduction) for _ in range(num_layers)
        ])

    def forward(self, features_list: list) -> torch.Tensor:
        #Apply channel-wise gating to each layer's features
        gated_features = [
            self.gated_modules[i](features_list[i]) for i in range(self.num_layers)
        ]
        
        #Concatenate the re-weighted features along the channel dimension
        return torch.cat(gated_features, dim = -1) #[B, P, C * num_layers]

# class GatedFeaturesFusion(nn.Module):
#     """
#     Fuses a list of feature tensors using a combination of learnable global weights
#     and an internal self-gating mechanism for each feature layer.
#     """
#     def __init__(self, num_layers: int, input_dim: int):
#         super().__init__()
#         #Global layer weights for each feature layer
#         self.global_layer_weights = nn.Parameter(torch.ones(num_layers)/num_layers)

#         #Self-gating mechanism for each feature layer
#         self.gating = nn.ModuleList([
#             nn.Linear(input_dim, input_dim) for _ in range(num_layers)
#         ])

#         #Normalization layer
#         self.norm = nn.LayerNorm(input_dim)

#     def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
#         #Normalize layer's weights
#         weights = torch.softmax(self.global_layer_weights, dim = 0)

#         #Gating and Weighted sum
#         fused_features = torch.zeros_like(features_list[0])
#         for i, features in enumerate(features_list):
#             #Self-gating mechanism: tank(W*x + b)*x
#             gated_features = torch.tanh(self.gating[i](features)) * features
#             fused_features += weights[i] * gated_features
        
#         return self.layer_norm(fused_features)
        


# class MeanGatedFeaturesFusion(nn.Module):
#     """
#     Fuses a list of feature tensors using a learned gating mechanism.
#     """
#     def __init__(self, num_layers: int, input_dim: int, hidden_dim: int):
#         super().__init__()
#         self.num_layers = num_layers
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_layers),
#             nn.Softmax(dim = -1)
#         )

#     #Forward pass for fusing features
#     def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
#         #[Batch*, Patches*, Dim] -> Stacked [Layers, Batch*, Patches*, Dim] -> Mean [Batch*, Patches*, Dim]
#         mean_feature = torch.stack(features_list, dim = 0).mean(dim = 0)

#         #Compute alpha gating weights
#         B, P, D = mean_feature.shape
#         mean_feature_flat = mean_feature.view(B * P, D) #[Batch*Patches, Dim]
#         alpha   = self.gate(mean_feature_flat)          #[Batch*Patches, num_layers]
#         alpha   = alpha.view(B, P, self.num_layers)     #[Batch*, Patches*, num_layers]

#         #Weighted sum of features: maintain shape [Batch*, Patches*, Dim]
#         gated = sum(
#             alpha[..., i].unsqueeze(-1) * features_list[i]
#             for i in range(self.num_layers)
#         )

#         return gated

# L = 5; D = 1024; H = 512
# fusion = MeanGatedFeaturesFusion(num_layers = L, input_dim = D, hidden_dim = H)

# # Giả sử có features_list với B=16, P=576
# features_list = [torch.randn(16, 576, D) for _ in range(L)]
# print(len(features_list), " in list of ", features_list[0].shape)
# fused_feature = fusion(features_list)
# print(fused_feature.shape)  # torch.Size([16, 576, 1024])
