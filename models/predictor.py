import torch.nn as nn

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        residual = self.linear1(x)
        out = self.linear2(self.relu(self.norm(residual)))
        return residual + out

class DTAPredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            ResidualMLP(embed_dim, embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        return self.predictor(x)
