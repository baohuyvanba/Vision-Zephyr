# =================================================================================================
# File: vis_zephyr/model/vision_encoder/gatingmlp.py
# Description: Implements a Gating MLP mechanism for fusing features from multiple layers.
# =================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseGatedSE(nn.Module):
    """
    Applies channel-wise Squeeze-and-Excitation and a GLU-like gate to a feature tensor.
    """
    def __init__(self, channel_dim, reduction=16):
        super().__init__()
        # Squeeze-and-Excitation part
        self.fc1 = nn.Linear(channel_dim, channel_dim // reduction, bias=False)
        self.fc2 = nn.Linear(channel_dim // reduction, channel_dim, bias=False)
        # Gated Linear Unit (GLU) part
        self.gate = nn.Linear(channel_dim, channel_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, C] - B: batch, P: patches, C: channel_dim
        B, P, C = x.shape
        
        # Squeeze: global average pooling across the patch dimension (P)
        squeeze = x.mean(dim=1)  # -> [B, C]
        
        # Excitation path
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))  # -> [B, C]
        
        # Gating path (GLU-like)
        gate_weights = torch.sigmoid(self.gate(squeeze))  # -> [B, C]
        
        # Combine weights and apply to the input tensor
        # The weights are broadcast across the patch dimension
        return x * (excitation * gate_weights).unsqueeze(1)

class CAGFRFusion(nn.Module):
    """
    Fuses a list of feature tensors by applying channel-wise gating to each
    and then concatenating the results.
    """
    def __init__(self, num_layers: int, channel_dim: int, reduction: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.channel_dim = channel_dim
        
        # Create a specific gating module for each feature layer
        self.gated_modules = nn.ModuleList([
            ChannelWiseGatedSE(channel_dim, reduction) for _ in range(num_layers)
        ])

    def forward(self, features_list: list) -> torch.Tensor:
        # Apply the corresponding gating module to each feature tensor in the list
        gated_features = [
            self.gated_modules[i](features_list[i]) for i in range(self.num_layers)
        ]
        
        # Concatenate the re-weighted features along the channel dimension
        # Output shape: [B, P, C * num_layers]
        return torch.cat(gated_features, dim=-1)






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
