import torch
import torch.nn as nn
from models.protein_encoder import ProteinGraphEncoder
from models.compound_encoder import CompoundEncoder
from models.attention_fusion import FeatureProjector, CrossAttention
from models.predictor import DTAPredictor

class DTARegressionModelNoFusion(nn.Module):
    def __init__(self, esm_dim=1280, cmp_feat_dim=384, protein_in_dim=38, compound_node_dim=78,
                 compound_edge_dim=6, hidden_dim=64, out_dim=128, dropout=0.1, num_heads=4, **kwargs):
        super().__init__()

        # 初始化蛋白质和药物编码器
        self.protein_encoder = ProteinGraphEncoder(protein_in_dim, hidden_dim, out_dim)
        self.compound_encoder = CompoundEncoder(compound_node_dim, compound_edge_dim, hidden_dim, out_dim)

        # ESM和药物特征投影器
        self.esm_proj = FeatureProjector(esm_dim, out_dim)
        self.cmp_feat_proj = FeatureProjector(cmp_feat_dim, out_dim)

        # 保留交叉注意力
        self.cross_attention_protein = CrossAttention(out_dim, out_dim, num_heads)
        self.cross_attention_compound = CrossAttention(out_dim, out_dim, num_heads)

        # 直接拼接蛋白质和药物的嵌入
        self.predictor = DTAPredictor(out_dim * 2)  # 拼接后的维度

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        full_graphs = batch['protein_full']
        compound_graphs = batch['compound']
        esm_vecs = batch['protein_esm']
        cmp_feat_vecs = batch['compound_feat']

        # 编码蛋白质和药物图
        full_embed = self.protein_encoder(full_graphs)
        compound_embed = self.compound_encoder(compound_graphs)
        esm_embed = self.esm_proj(esm_vecs)
        cmp_feat_embed = self.cmp_feat_proj(cmp_feat_vecs)

        # 使用交叉注意力
        full_embed = self.cross_attention_protein(full_embed.unsqueeze(1), esm_embed.unsqueeze(1)).squeeze(1)
        compound_embed = self.cross_attention_compound(compound_embed.unsqueeze(1), cmp_feat_embed.unsqueeze(1)).squeeze(1)

        # 直接拼接蛋白质和药物嵌入
        fusion = torch.cat([full_embed, compound_embed], dim=-1)

        # 进行预测
        out = self.predictor(fusion)
        return out
