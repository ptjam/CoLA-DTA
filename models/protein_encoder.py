import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class ProteinGraphEncoder(nn.Module):
    def __init__(self, in_dim=38, hidden_dim=64, out_dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1, dropout=dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.gat2(x, edge_index))
        return global_mean_pool(x, batch)
