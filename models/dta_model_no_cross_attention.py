import torch
import torch.nn as nn
from models.protein_encoder import ProteinGraphEncoder
from models.compound_encoder import CompoundEncoder
from models.attention_fusion import FeatureProjector, MultiHeadGatedFusion
from models.predictor import DTAPredictor

class DTARegressionModelNoCrossAttention(nn.Module):
    def __init__(self, esm_dim=1280, cmp_feat_dim=384, protein_in_dim=38, compound_node_dim=78,
                 compound_edge_dim=6, hidden_dim=64, out_dim=128, dropout=0.1, num_heads=4, **kwargs):
        super().__init__()

        # 接受从配置文件传递来的超参数
        self.protein_encoder = ProteinGraphEncoder(protein_in_dim, hidden_dim, out_dim)
        self.compound_encoder = CompoundEncoder(compound_node_dim, compound_edge_dim, hidden_dim, out_dim)

        self.esm_proj = FeatureProjector(esm_dim, out_dim)
        self.cmp_feat_proj = FeatureProjector(cmp_feat_dim, out_dim)

        # 去除 CrossAttention，改为 Gated Fusion
        self.fuse_protein = MultiHeadGatedFusion(out_dim, num_heads)
        self.fuse_compound = MultiHeadGatedFusion(out_dim, num_heads)

        # 定义预测器
        self.predictor = DTAPredictor(out_dim * 2)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        # 获取输入数据
        full_graphs = batch['protein_full']
        compound_graphs = batch['compound']
        esm_vecs = batch['protein_esm']
        cmp_feat_vecs = batch['compound_feat']

        # 编码蛋白质和药物图
        full_embed = self.protein_encoder(full_graphs)
        compound_embed = self.compound_encoder(compound_graphs)
        esm_embed = self.esm_proj(esm_vecs)
        cmp_feat_embed = self.cmp_feat_proj(cmp_feat_vecs)

        # 融合蛋白质和药物嵌入
        fused_protein = self.fuse_protein(full_embed, esm_embed)
        fused_compound = self.fuse_compound(compound_embed, cmp_feat_embed)

        # 拼接蛋白质和药物的特征
        fusion = torch.cat([fused_protein, fused_compound], dim=-1)

        # 最终预测
        out = self.predictor(fusion)
        return out
