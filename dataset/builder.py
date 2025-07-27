import os
from dataset import CPIDataset

class DatasetBuilder:
    def __init__(self, split_manager, protein_graph_dir, compound_graph_dir, protein_feat_pkl, compound_feat_pkl, preload=True):
        """
        Dataset 构建器，根据划分模式与 DataFrame 创建最终训练与测试数据集

        参数:
            split_manager: SplitManager 实例 (已完成数据划分与 DataFrame 生成)
            protein_graph_dir: 蛋白质图目录
            compound_graph_dir: 化合物图目录
            protein_feat_pkl: 蛋白质特征文件
            compound_feat_pkl: 化合物特征文件
            preload: 是否预加载数据集到内存中（默认为False）
        """
        self.split_manager = split_manager
        self.protein_graph_dir = protein_graph_dir
        self.compound_graph_dir = compound_graph_dir
        self.protein_feat_pkl = protein_feat_pkl
        self.compound_feat_pkl = compound_feat_pkl
        self.preload = preload

    def build(self):
        """
        返回训练集与测试集 Dataset

        - standard 模式下返回: (train_dataset, val_dataset, test_dataset)
        - cold-start 模式下返回: (train_dataset, dict_of_test_datasets)
        """
        train_df, val_df, test_df = self.split_manager.get_split()

        # 创建训练集、验证集、测试集
        train_dataset = self._create_dataset(train_df)
        val_dataset = self._create_dataset(val_df)
        test_dataset = self._create_dataset(test_df)

        if self.split_manager.mode == 'standard':
            return train_dataset, val_dataset, test_dataset
        elif self.split_manager.mode == 'cold-start':
            # 处理 cold-start 模式下的多个测试集
            test_datasets = {
                name: self._create_dataset(df)
                for name, df in test_df.items()
            }
            return train_dataset, test_datasets
        else:
            raise ValueError(f"Unsupported mode: {self.split_manager.mode}")

    def _create_dataset(self, dataframe):
        """
        内部函数：DataFrame 转为 CPIDataset
        """
        return CPIDataset(
            dataframe=dataframe,
            protein_graph_dir=self.protein_graph_dir,
            compound_graph_dir=self.compound_graph_dir,
            protein_feat_pkl_path=self.protein_feat_pkl,
            compound_feat_pkl_path=self.compound_feat_pkl,
            preload=self.preload  # 使用传入的 preload 设置
        )
