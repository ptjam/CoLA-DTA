import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class CompoundEncoder(nn.Module):
    def __init__(self, node_dim=78, edge_dim=6, hidden_dim=128, out_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gnn_layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.gnn_layers.append(GINEConv(nn=mlp, edge_dim=edge_dim))

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        for conv in self.gnn_layers:
            x = F.relu(conv(x, edge_index, edge_attr))
        x = global_mean_pool(x, graph.batch)
        return self.proj(x)
