# =================================================================================================
# File: vis_zephyr/model/vision_encoder/gatingmlp.py
# Description: Implements a Gating MLP mechanism for fusing features from multiple layers.
# =================================================================================================
from curses import start_color
from click import group
from numpy import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseChannelIntegrationFusion(nn.Module):
    """
    """
    def __init__(self, num_groups: int = 4):
        super().__init__()
        if num_groups < 0:
            raise ValueError("Number of groups must be a positive integer.")
        self.num_groups = num_groups

    def forward(self, features_list: list) -> torch.Tensor:
        """ """
        if len(features_list) < self.num_groups + 1:
            raise ValueError(f"Expected at least {self.num_groups + 1} feature tensors (including each feature for each group and final layer), got {len(features_list)}.")
        
        final_features_layer  =  features_list[-1]  # Last layer features
        intermediate_features = features_list[:-1]  # All but the last layer features
        num_intermediate      = len(intermediate_features)

        if num_intermediate % self.num_groups != 0:
            raise ValueError(f"Number of intermediate features ({num_intermediate}) must be divisible by num_groups ({self.num_groups}).")
        
        layers_per_group = num_intermediate // self.num_groups
        fused_features_group = []
        for i in range(self.num_groups):
            start_idx = i * layers_per_group
            end_inx   = start_idx + layers_per_group

            group_features = intermediate_features[start_idx:end_inx]

            stacked_group_features = torch.stack(group_features, dim = 0)      # [G, B, P, C]
            mean_group_features    = torch.mean(stacked_group_features, dim=0) # [B, P, C]
            
            fused_features_group.append(mean_group_features)
        
        all_fused_features = fused_features_group + [final_features_layer]  # [B, P, C]*5
        concatenated_features = torch.cat(all_fused_features, dim = -1)     # [B, P, C * 5]

        return concatenated_features

# METHOD 4: Concat -> MLP
# class MultiLayerFeatureFusionMLP(nn.Module):
#     """
#     Fusion 5 lớp đặc trưng bằng MLP: ghép nối theo chiều kênh và đưa qua hai lớp Linear.
#     Kết quả: [B, P, C * num_layers].
#     """
#     def __init__(self, num_layers: int, channel_dim: int):
#         super().__init__()
#         self.num_layers = num_layers
#         self.channel_dim = channel_dim
#         # MLP: đầu vào C*num_layers, ẩn và đầu ra cùng C*num_layers
#         hidden_dim = channel_dim * num_layers
#         self.fc = nn.Sequential(
#             nn.Linear(channel_dim * num_layers, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, channel_dim * num_layers)
#         )

#     def forward(self, features_list: list) -> torch.Tensor:
#         # features_list gồm N tensor [B,P,C]
#         B, P, C = features_list[0].shape
#         # 1. Ghép nối theo chiều kênh -> [B, P, C*num_layers]
#         x = torch.cat(features_list, dim=-1)
#         # 2. Áp dụng MLP từng patch: chuyển về [B*P, C*N]
#         x = x.view(B*P, self.channel_dim * self.num_layers)
#         x = self.fc(x)            # [B*P, C*N]
#         x = x.view(B, P, self.channel_dim * self.num_layers)

# METHOD 3:
# class ChannelWiseGatedFusion(nn.Module):
#     """
#     Applies channel-wise feature reweighting using: 
#       - Squeeze-and-Excitation (SE);
#       - GLU-inspired gating mechanism.
#     """
#     def __init__(self, channel_dim, reduction=16):
#         super().__init__()
#         reduced_dim = channel_dim // reduction

#         #Squeeze-and-Excitation (SE) layers (MLP layers)
#         self.fc1 = nn.Linear(channel_dim, reduced_dim, bias = False)
#         self.fc2 = nn.Linear(reduced_dim, channel_dim, bias = False)
        
#         #Gated Linear Unit (GLU) layer: channel's importance
#         self.gate = nn.Linear(channel_dim, channel_dim, bias = False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         #x: [B, P, D] - B: batch, P: patches, D: dim
#         B, P, C = x.shape
        
#         #1. Global average pooling across the patch dimension (P) -> Global context: [B, D]
#         pooled_patch = x.mean(dim = 1)
        
#         #2. SE path
#         se = F.relu(self.fc1(pooled_patch))  # -> [D, D//r]
#         se = torch.sigmoid(self.fc2(se))     # -> [D//r, D]
        
#         #3. Gating path 
#         gate_weights = torch.sigmoid(self.gate(pooled_patch))  # -> [B, D]
        
#         #4. Combine weights and apply to the input tensor
#         combined_weight = (se * gate_weights).unsqueeze(1)     # -> [B, 1, D]

#         return x * combined_weight

# class MultiLayerFeatureFusion(nn.Module):
#     """
#     Fuses multiple feature tensors (from different layers) using channel-wise gated fusion
#     -> Concatenates the results along the channel dimension.
#     """
#     def __init__(self, num_layers: int, channel_dim: int, reduction: int = 16):
#         super().__init__()
#         self.num_layers  = num_layers
#         self.channel_dim = channel_dim
        
#         # ChannelWiseGatedFusion module ~ each input feature
#         self.gated_modules = nn.ModuleList([
#             ChannelWiseGatedFusion(channel_dim, reduction) for _ in range(num_layers)
#         ])

#     def forward(self, features_list: list) -> torch.Tensor:
#         #Apply channel-wise gating to each layer's features
#         gated_features = [
#             self.gated_modules[i](features_list[i]) for i in range(self.num_layers)
#         ]
        
#         #Concatenate the re-weighted features along the channel dimension
#         return torch.cat(gated_features, dim = -1) #[B, P, C * num_layers]

# METHOD 2:
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
        
# METHOD 1: Mean
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
