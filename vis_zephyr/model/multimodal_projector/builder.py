from mimetypes import init
import torch
import torch.nn as nn
import re

#====================================================================================================================================
# QFormer Block Definition
#====================================================================================================================================
class QFormerBlock(nn.Module):
    def __init__(self, hidden_size, nhead, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            num_heads=nhead,
            embed_dim=4096,
            kdim=5120,
            vdim=5120,
            batch_first=True
        )
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
    
#====================================================================================================================================
# QFormer Definition
#====================================================================================================================================
class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_queries = 16
        self.hidden_size = config.hidden_size
        
        self.learned_queries = nn.Parameter(torch.randn(self.num_queries, self.hidden_size))
        self.blocks = nn.ModuleList([
            QFormerBlock(
                hidden_size = self.hidden_size,
                nhead       = 8, #num_heads
                ffn_dim     = 4096
            )
            for _ in range(5)
        ])

        self.pre_norm = nn.LayerNorm(5120)
        self.norm     = nn.LayerNorm(self.hidden_size)

    def forward(self, features, text_embeddings=None):
        B        = features.size(0)
        features = self.pre_norm(features) 

        queries  = self.learned_queries.unsqueeze(0).expand(B, -1, -1)
        
        if text_embeddings is not None:
            if isinstance(text_embeddings, list):
                text_embeddings = torch.stack(text_embeddings, dim=0)  # [B, L, D]
                if text_embeddings.shape[0] < B:
                    repeat_times = (B + text_embeddings.shape[0] - 1) // text_embeddings.shape[0]
                    text_embeddings = text_embeddings.repeat((repeat_times, 1, 1))[:B]

            if queries.shape[1] == 1:  # <== chỉ concat nếu chưa concat
                queries = torch.cat([queries, text_embeddings], dim=1)  # [B, Q+L, D]

        for blk in self.blocks:
            queries = blk(queries, features)

        return self.norm(queries[:, :self.num_queries, :])

#====================================================================================================================================
# Main Multimodal-Projector Builder
#====================================================================================================================================
def build_multimodal_projector(config, **kwargs):
    """
    Build the multimodal projector based on the provided configuration
    """
    return QFormer(config)
