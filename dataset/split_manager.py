import os
import pandas as pd

class SplitManager:
    def __init__(self, mode, dataset_dir):
        """
        :param mode: 'standard' 或 'cold-start'
        :param dataset_dir: 数据集目录（如 data/PDB）
        """
        self.mode = mode
        self.dataset_dir = dataset_dir

    def get_split(self):
        if self.mode == 'standard':
            return self._standard_split()
        elif self.mode == 'cold-start':
            return self._cold_start_split()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _standard_split(self):
        """
        standard 模式：直接读取 standard 目录下的 train/val/test.tsv
        如果 test.tsv 不存在，将 val 作为 test 使用
        """
        standard_dir = os.path.join(self.dataset_dir, 'standard')

        train_path = os.path.join(standard_dir, 'train.tsv')
        val_path = os.path.join(standard_dir, 'val.tsv')
        test_path = os.path.join(standard_dir, 'test.tsv')

        # 文件存在性检查
        for file in [train_path, val_path]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"找不到文件: {file}")

        train_df = pd.read_csv(train_path, sep='\t')
        val_df = pd.read_csv(val_path, sep='\t')

        # 如果 test.tsv 存在则读取，否则使用 val 作为测试集
        test_df = None
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path, sep='\t')
        else:
            test_df = val_df  # 如果没有 test.tsv，则使用 val 作为测试集

        # 确保数据框中的 PDB_ID 和 compound_id 是字符串类型
        for df in [train_df, val_df, test_df]:
            df['PDB_ID'] = df['PDB_ID'].astype(str)
            df['compound_id'] = df['compound_id'].astype(str)

        return train_df, val_df, test_df

    def _cold_start_split(self):
        """
        cold-start 模式：直接读取划分好的训练集 + 3个子测试集
        """
        split_dir = os.path.join(self.dataset_dir, 'cold_start')

        train_path = os.path.join(split_dir, 'train.tsv')
        test_all_path = os.path.join(split_dir, 'cold_all.tsv')
        test_protein_path = os.path.join(split_dir, 'cold_protein.tsv')
        test_drug_path = os.path.join(split_dir, 'cold_drug.tsv')

        for file in [train_path, test_all_path, test_protein_path, test_drug_path]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"找不到文件: {file}")

        train_df = pd.read_csv(train_path, sep='\t')
        test_all_df = pd.read_csv(test_all_path, sep='\t')
        test_protein_df = pd.read_csv(test_protein_path, sep='\t')
        test_drug_df = pd.read_csv(test_drug_path, sep='\t')

        # 确保数据框中的 PDB_ID 和 compound_id 是字符串类型
        for df in [train_df, test_all_df, test_protein_df, test_drug_df]:
            df['PDB_ID'] = df['PDB_ID'].astype(str)
            df['compound_id'] = df['compound_id'].astype(str)

        return train_df, {
            'cold_all': test_all_df.reset_index(drop=True),
            'cold_protein': test_protein_df.reset_index(drop=True),
            'cold_drug': test_drug_df.reset_index(drop=True)
        }
