import torch
import torch.nn as nn
from models.compound_encoder import CompoundEncoder
from models.attention_fusion import FeatureProjector, CrossAttention
from models.predictor import DTAPredictor

class DTARegressionModelNoProteinGraph(nn.Module):
    def __init__(self, esm_dim=1280, cmp_feat_dim=384, protein_in_dim=0, compound_node_dim=78,
                 compound_edge_dim=6, hidden_dim=64, out_dim=128, dropout=0.1, num_heads=4, **kwargs):
        super().__init__()

        # 只使用蛋白质的序列嵌入，不使用蛋白质图
        self.esm_proj = FeatureProjector(esm_dim, out_dim)
        self.cmp_feat_proj = FeatureProjector(cmp_feat_dim, out_dim)

        # 只保留化合物图编码器
        self.compound_encoder = CompoundEncoder(compound_node_dim, compound_edge_dim, hidden_dim, out_dim)

        # 保留交叉注意力
        self.cross_attention_protein = CrossAttention(out_dim, out_dim, num_heads)
        self.cross_attention_compound = CrossAttention(out_dim, out_dim, num_heads)

        # 预测器
        self.predictor = DTAPredictor(out_dim * 2)  # 拼接后的维度

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        compound_graphs = batch['compound']
        esm_vecs = batch['protein_esm']
        cmp_feat_vecs = batch['compound_feat']

        # 只处理蛋白质的序列嵌入，不处理蛋白质图
        esm_embed = self.esm_proj(esm_vecs)
        cmp_feat_embed = self.cmp_feat_proj(cmp_feat_vecs)

        # 编码化合物图
        compound_embed = self.compound_encoder(compound_graphs)

        # 使用交叉注意力
        full_embed = self.cross_attention_protein(esm_embed.unsqueeze(1), esm_embed.unsqueeze(1)).squeeze(1)
        compound_embed = self.cross_attention_compound(compound_embed.unsqueeze(1), cmp_feat_embed.unsqueeze(1)).squeeze(1)

        # 直接拼接蛋白质和药物的特征
        fusion = torch.cat([full_embed, compound_embed], dim=-1)

        # 最终预测
        out = self.predictor(fusion)
        return out
