import torch
import torch.nn as nn
from models.protein_encoder import ProteinGraphEncoder
from models.compound_encoder import CompoundEncoder
from models.attention_fusion import FeatureProjector, CrossAttention, MultiHeadGatedFusion
from models.predictor import DTAPredictor

class DTARegressionModel(nn.Module):
    def __init__(
        self,
        esm_dim=1280,
        cmp_feat_dim=384,
        protein_in_dim=38,
        compound_node_dim=78,
        compound_edge_dim=6,
        hidden_dim=64,
        out_dim=128,
        dropout=0.1,
        num_heads=4,
    ):
        super().__init__()

        self.protein_encoder = ProteinGraphEncoder(protein_in_dim, hidden_dim, out_dim)
        self.compound_encoder = CompoundEncoder(compound_node_dim, compound_edge_dim, hidden_dim, out_dim)

        self.esm_proj = FeatureProjector(esm_dim, out_dim)
        self.cmp_feat_proj = FeatureProjector(cmp_feat_dim, out_dim)

        self.cross_full_esm = CrossAttention(out_dim, out_dim, num_heads)
        self.cross_protein_compound = CrossAttention(out_dim, out_dim, num_heads)
        self.cross_compound_protein = CrossAttention(out_dim, out_dim, num_heads)

        self.fuse_protein = MultiHeadGatedFusion(out_dim, num_heads)
        self.fuse_compound = MultiHeadGatedFusion(out_dim, num_heads)

        self.predictor = DTAPredictor(out_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        full_graphs = batch['protein_full']
        compound_graphs = batch['compound']
        esm_vecs = batch['protein_esm']
        cmp_feat_vecs = batch['compound_feat']

        full_embed = self.protein_encoder(full_graphs)
        compound_embed = self.compound_encoder(compound_graphs)
        esm_embed = self.esm_proj(esm_vecs)
        cmp_feat_embed = self.cmp_feat_proj(cmp_feat_vecs)

        full_embed = self.cross_full_esm(full_embed.unsqueeze(1), esm_embed.unsqueeze(1)).squeeze(1)
        full_embed = self.dropout(full_embed)
        fused_protein = self.fuse_protein(full_embed, esm_embed)
        compound_embed = self.fuse_compound(compound_embed, cmp_feat_embed)

        joint_protein = self.cross_protein_compound(fused_protein.unsqueeze(1), compound_embed.unsqueeze(1)).squeeze(1)
        joint_protein = self.dropout(joint_protein)
        joint_compound = self.cross_compound_protein(compound_embed.unsqueeze(1), fused_protein.unsqueeze(1)).squeeze(1)
        joint_compound = self.dropout(joint_compound)

        fusion = torch.cat([joint_protein, joint_compound], dim=-1)
        out = self.predictor(fusion)
        return out
