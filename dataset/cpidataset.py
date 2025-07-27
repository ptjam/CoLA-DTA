import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

class CPIDataset(Dataset):
    def __init__(self, tsv_file, protein_graph_dir, compound_graph_dir,
                 protein_feat_pkl_path, compound_feat_pkl_path, preload=True):
        super().__init__()

        # 读取 TSV 文件
        self.df = pd.read_csv(tsv_file, sep='\t')  # 假设分隔符为制表符（tab）
        self.df['PDB_ID'] = self.df['PDB_ID'].astype(str)
        self.df['compound_id'] = self.df['compound_id'].astype(str)
        self.df['label'] = self.df['label'].astype(float)  # 确保标签是float类型

        self.protein_graph_dir = protein_graph_dir
        self.compound_graph_dir = compound_graph_dir

        # 读取蛋白质和化合物特征字典
        self.protein_feat_dict = torch.load(protein_feat_pkl_path, weights_only=False)
        self.compound_feat_dict = torch.load(compound_feat_pkl_path, weights_only=False)

        self.preload = preload
        self.memory_cache = {}

        if self.preload:
            # 去掉这里的 print 语句
            self._preload_all()

    def _preload_all(self):
        """加载所有数据到内存缓存中，并添加 tqdm 进度条"""
        # 使用 tqdm 进行进度条显示
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preloading samples", mininterval=60.0):
            self.memory_cache[idx] = self._load_sample(row)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 如果启用预加载，则直接返回缓存中的样本
        if self.preload:
            return self.memory_cache[idx]
        else:
            row = self.df.iloc[idx]
            return self._load_sample(row)

    def _load_sample(self, row):
        pdb_id = row['PDB_ID']
        compound_id = row['compound_id']
        label = row['label']

        # 加载蛋白质图
        protein_graph_path = os.path.join(self.protein_graph_dir, f"{pdb_id}.pt")
        protein_graph = torch.load(protein_graph_path, weights_only=False)  # 直接加载 Data
        if not isinstance(protein_graph, Data):
            protein_graph = Data(**protein_graph)

        # 加载化合物图
        compound_graph_path = os.path.join(self.compound_graph_dir, f"{compound_id}.pt")
        compound_graph = torch.load(compound_graph_path, weights_only=False)  # 直接加载 Data
        if not isinstance(compound_graph, Data):
            compound_graph = Data(**compound_graph)

        # 加载特征
        protein_esm = torch.tensor(self.protein_feat_dict[pdb_id], dtype=torch.float)
        compound_feat = torch.tensor(self.compound_feat_dict[compound_id], dtype=torch.float)

        # 使用 Data 类型返回
        data = Data(
            protein_full=protein_graph,
            compound=compound_graph,
            protein_esm=protein_esm,
            compound_feat=compound_feat,
            label=torch.tensor(label, dtype=torch.float)
        )

        # 显式添加 protein_id 和 compound_id
        data.protein_id = pdb_id  # 添加 protein_id 字段
        data.compound_id = compound_id  # 添加 compound_id 字段

        return data


# 自定义collate函数
def collate_fn(batch_list):
    # 使用 Batch.from_data_list 合并 protein_full 和 compound
    protein_full_batch = Batch.from_data_list([s['protein_full'] for s in batch_list])
    compound_batch = Batch.from_data_list([s['compound'] for s in batch_list])
    
    # 获取 protein_id 和 compound_id
    protein_ids = [s['protein_id'] for s in batch_list]
    compound_ids = [s['compound_id'] for s in batch_list]
    
    # 合并特征和标签
    protein_esm_batch = torch.stack([s['protein_esm'] for s in batch_list])
    compound_feat_batch = torch.stack([s['compound_feat'] for s in batch_list])
    label_batch = torch.stack([s['label'] for s in batch_list])

    # 返回字典，包含所有信息（图数据和非图数据）
    return {
        'protein_full': protein_full_batch,
        'compound': compound_batch,
        'protein_esm': protein_esm_batch,
        'compound_feat': compound_feat_batch,
        'label': label_batch,
        'protein_id': protein_ids,  # 返回 protein_id 列表
        'compound_id': compound_ids  # 返回 compound_id 列表
    }


if __name__ == "__main__":
    # 假设 TSV 文件路径
    train_tsv_file = '../data/DAVIS/standard/train.tsv'  # 替换为你的TSV文件路径
    val_tsv_file = '../data/DAVIS/standard/val.tsv'  # 替换为你的验证集TSV文件路径
    test_tsv_file = '../data/DAVIS/standard/test.tsv'  # 或者 val_tsv_file，根据需要选择
    protein_graph_dir = '../data/DAVIS/protein_graph'
    compound_graph_dir = '../data/DAVIS/compound_graphs'
    protein_feat_pkl_path = '../data/DAVIS/protein_esm_feature.pkl'
    compound_feat_pkl_path = '../data/DAVIS/compound_bert_features.pkl'

    # 创建数据集实例
    train_dataset = CPIDataset(train_tsv_file, protein_graph_dir, compound_graph_dir, protein_feat_pkl_path, compound_feat_pkl_path, preload=False)
    val_dataset = CPIDataset(val_tsv_file, protein_graph_dir, compound_graph_dir, protein_feat_pkl_path, compound_feat_pkl_path, preload=False)
    test_dataset = CPIDataset(test_tsv_file, protein_graph_dir, compound_graph_dir, protein_feat_pkl_path, compound_feat_pkl_path, preload=False)

    # 测试getitem
    sample = train_dataset[0]
    print(f"Sample 0: {sample}")

    # 创建数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

    # 打印批次
    for batch in train_data_loader:
        print(f"Batch: {batch}")
        break
