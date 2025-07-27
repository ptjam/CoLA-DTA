import torch
import torch.nn as nn
from models.protein_encoder import ProteinGraphEncoder
from models.attention_fusion import FeatureProjector, CrossAttention, MultiHeadGatedFusion
from models.predictor import DTAPredictor



class DTARegressionModelWithoutCompoundGraph(nn.Module):
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

        # Protein encoder
        self.protein_encoder = ProteinGraphEncoder(protein_in_dim, hidden_dim, out_dim)

        # ESM projection
        self.esm_proj = FeatureProjector(esm_dim, out_dim)
        self.cmp_feat_proj = FeatureProjector(cmp_feat_dim, out_dim)

        # Cross attention and fusion mechanisms
        self.cross_full_esm = CrossAttention(out_dim, out_dim, num_heads)
        self.cross_protein_compound = CrossAttention(out_dim, out_dim, num_heads)
        self.cross_compound_protein = CrossAttention(out_dim, out_dim, num_heads)
        self.fuse_protein = MultiHeadGatedFusion(out_dim, num_heads)

        # Final predictor
        self.predictor = DTAPredictor(out_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        full_graphs = batch['protein_full']
        compound_feat_vecs = batch['compound_feat']  # Keep compound features
        esm_vecs = batch['protein_esm']

        # Protein encoding
        full_embed = self.protein_encoder(full_graphs)
        # print(f"full_embed shape: {full_embed.shape}")

        # ESM projection
        esm_embed = self.esm_proj(esm_vecs)
        # print(f"esm_embed shape: {esm_embed.shape}")

        # Compound feature projection (keep the compound features)
        cmp_feat_embed = self.cmp_feat_proj(compound_feat_vecs)
        # print(f"cmp_feat_embed shape: {cmp_feat_embed.shape}")

        # Cross attention between protein and ESM
        full_embed = self.cross_full_esm(full_embed.unsqueeze(1), esm_embed.unsqueeze(1)).squeeze(1)
        # print(f"full_embed after cross_full_esm shape: {full_embed.shape}")
        full_embed = self.dropout(full_embed)

        # Fusion between protein and ESM
        fused_protein = self.fuse_protein(full_embed, esm_embed)
        # print(f"fused_protein shape: {fused_protein.shape}")

        # Cross attention between fused protein and compound features
        joint_protein_compound = self.cross_protein_compound(fused_protein.unsqueeze(1), cmp_feat_embed.unsqueeze(1)).squeeze(1)
        joint_compound_protein = self.cross_compound_protein(cmp_feat_embed.unsqueeze(1), fused_protein.unsqueeze(1)).squeeze(1)
        # print(f"joint_protein_compound shape: {joint_protein_compound.shape}")
        joint_protein_compound = self.dropout(joint_protein_compound)
        joint_compound_protein = self.dropout(joint_compound_protein)

        # Final prediction
        fusion = torch.cat([joint_protein_compound, joint_compound_protein], dim=-1)
        # print(f"fusion shape: {fusion.shape}")
        out = self.predictor(fusion)
        return out


