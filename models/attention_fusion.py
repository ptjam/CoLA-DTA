import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim_q=128, dim_kv=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, kdim=dim_kv, vdim=dim_kv, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_q)
        self.proj = nn.Sequential(
            nn.Linear(dim_q, dim_q),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, query, context):
        attn_out, _ = self.attn(query, context, context)
        proj = self.proj(attn_out)
        return self.norm(query + proj)

class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x)

class MultiHeadGatedFusion(nn.Module):
    def __init__(self, dim, num_heads=4, use_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim)

    def forward(self, a, b):
        outs = []
        for linear in self.linears:
            gate = linear(torch.cat([a, b], dim=-1))
            fused = gate * a + (1 - gate) * b + a
            outs.append(fused)
        out = torch.mean(torch.stack(outs, dim=0), dim=0)
        return self.norm(out) if self.use_norm else out
