# =================================================================================================
# File: vis_zephyr/model/multimodal_projector/builder.py
# Description: Builder for the multimodal projector component of the Vision-Zephyr model.
#              This component is responsible for processing visual features and integrating them with text embeddings.
# =================================================================================================
import torch
import torch.nn as nn

#====================================================================================================================================
# QFormer Block Definition
#====================================================================================================================================
class QFormerBlock(nn.Module):
    def __init__(self, hidden_size, nhead, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            num_heads   = nhead,
            embed_dim   = 4096,
            kdim        = 5120,
            vdim        = 5120,
            batch_first = True
        )
        
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
    
#====================================================================================================================================
# QFormer Definition
#====================================================================================================================================
class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_queries = 32
        self.hidden_size = config.hidden_size
        self.learned_queries = nn.Parameter(torch.randn(self.num_queries, self.hidden_size))
        
        num_blocks  = 8 
        ffn_dim     = self.hidden_size * 2
        self.blocks = nn.ModuleList([
            QFormerBlock(
                hidden_size = self.hidden_size,
                nhead       = 8, #num_heads
                ffn_dim     = ffn_dim #4096
            )
            for _ in range(num_blocks)
        ])

        #Visual features normalization
        self.pre_norm = nn.LayerNorm(5120)
        #Final normalization
        self.norm     = nn.LayerNorm(self.hidden_size)

    def forward(self, features, text_embeddings=None):
        Batch_size = features.size(0)
        features   = self.pre_norm(features)
        queries    = self.learned_queries.unsqueeze(0).expand(Batch_size, -1, -1)
        
        if text_embeddings is not None:
            init_queries = torch.cat([queries, text_embeddings], dim = 1)  # [B, Q+L, D]
        else:
            init_queries = queries

        # for blk in self.blocks:
        #     queries = blk(queries, features)
        processed_inputs = self.blocks[0](
            init_queries, features
        )
        queries = processed_inputs[:, :self.num_queries, :]
        
        for blk in self.blocks[1:]:
            queries = blk(queries, features)

        return self.norm(queries)

#====================================================================================================================================
# Main Multimodal-Projector Builder
#====================================================================================================================================
def build_multimodal_projector(config, **kwargs):
    """
    Build the multimodal projector based on the provided configuration
    """
    return QFormer(config)
