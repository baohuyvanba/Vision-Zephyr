from mimetypes import init
import torch
import torch.nn as nn
import re

class QFormerBlock(nn.Module):
    def __init__(self, hidden_size, nhead, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_size),
        )

    def forward(self, queries, visual_features):
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, visual_features, visual_features)[0]
        q = self.norm3(queries)
        queries = queries + self.ffn(q)
        return queries


class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_queries = config.num_projector_queries
        self.hidden_size = config.hidden_size

        self.learned_queries = nn.Parameter(torch.randn(self.num_queries, self.hidden_size))
        self.blocks = nn.ModuleList([
            QFormerBlock(
                hidden_size=self.hidden_size,
                nhead=config.projector_nhead,
                ffn_dim=config.projector_ffn_dim
            )
            for _ in range(config.projector_num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, features, text_embeddings=None):
        B = features.size(0)
        queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1)

        if text_embeddings is not None:
            queries = torch.cat([queries, text_embeddings], dim=1)

        for blk in self.blocks:
            queries = blk(queries, features)

        return self.norm(queries[:, :self.num_queries, :])

class SimpleFeatureSingleModel(nn.Module):
    """
    Apply Layer-Normalization to the input features, before passing them through MLP layers.
    """
    def __init__(self, num_clip_layers, final_linear):
        super(SimpleFeatureSingleModel, self).__init__()
        self.clip_layer_norm = nn.LayerNorm(num_clip_layers)
        self.final_linear    = final_linear
    
    def forward(self, features, text_embeddings):
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

    projector_type = 'qformer'

    if projector_type == 'qformer':
        return QFormer(config)

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

