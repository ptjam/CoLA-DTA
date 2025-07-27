import os
import torch
import pandas as pd
from tqdm import tqdm
from tape import ProteinBertModel, TAPETokenizer
import numpy as np

class ProteinFeatureExtractor:
    def __init__(self, data_dir, output_dir, model_name='bert-base', vocab='iupac', batch_size=1):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.vocab = vocab
        self.batch_size = batch_size  # 设置批次大小
        
        # 初始化模型和tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ProteinBertModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = TAPETokenizer(vocab=self.vocab)
        print(f"Using model: {self.model_name} with vocab: {self.vocab}")
        
    def process_datasets(self):
        """
        遍历 /data 下的每个数据集文件夹，处理 each all.tsv 文件，并生成 protein features。
        """
        dataset_folders = ['BDB', 'DAVIS', 'PDB', 'KIBA']

        for dataset_folder in dataset_folders:
            dataset_path = os.path.join(self.data_dir, dataset_folder)
            if os.path.isdir(dataset_path):
                tsv_path = os.path.join(dataset_path, 'all.tsv')
                if os.path.exists(tsv_path):
                    print(f"Processing {dataset_folder}...")

                    # 使用 tqdm 显示每个数据集的进度条
                    df = pd.read_csv(tsv_path, sep='\t')
                    print(f"Loaded {len(df)} rows from {dataset_folder}/all.tsv")
                    
                    # 查找包含蛋白质序列的列
                    sequence_column = None
                    for col in df.columns:
                        if 'sequence' in col.lower():  # 查找列名中包含 'sequence' 的列，不区分大小写
                            sequence_column = col
                            break

                    if sequence_column is None:
                        print(f"Sequence column not found in {dataset_folder}/all.tsv")
                        continue  # 如果没有找到序列列，则跳过该数据集

                    # 对 sequence 列进行去重
                    df = df.drop_duplicates(subset=[sequence_column])  # 去掉重复的氨基酸序列

                    all_features = {}
                    # 使用批量处理
                    for start_idx in tqdm(range(0, len(df), self.batch_size), desc=f"{dataset_folder} Processing"):
                        batch = df.iloc[start_idx:start_idx + self.batch_size]
                        batch_features = self.process_batch(batch, sequence_column)
                        all_features.update(batch_features)
                    
                    # 将特征保存为 pkl 文件
                    self.save_features_to_pkl(all_features, dataset_folder)
                else:
                    print(f"{dataset_folder}/all.tsv not found. Skipping this dataset.")

    def process_batch(self, batch, sequence_column):
        """
        批量处理蛋白质序列，提取特征。
        """
        batch_features = {}
        for _, row in batch.iterrows():
            pdb_id = str(row['PDB_ID'])
            sequence = row[sequence_column]  # 使用找到的列名来获取序列
            
            # 获取蛋白质特征
            feature = self.extract_protein_features(sequence)
            batch_features[pdb_id] = feature
        return batch_features

    def extract_protein_features(self, sequence):
        """
        使用 TAPE 提取蛋白质序列的特征。
        """
        sequence = sequence.upper()  # 确保所有字母为大写
        allowed_characters = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = ''.join([c for c in sequence if c in allowed_characters])

        # 截取前512个氨基酸
        sequence = sequence[:512]

        # 编码蛋白质序列
        token_ids = torch.tensor(np.array([self.tokenizer.encode(sequence)])).to(self.device)

        # 使用 no_grad() 避免梯度计算
        with torch.no_grad():
            output = self.model(token_ids)
            sequence_output = output[0]  # 获取序列输出

        # 截取 [CLS] 和 [SEP] 标记
        sequence_output = sequence_output[:, 1:-1]  # 去掉 [CLS] 和 [SEP]，保留实际的序列特征

        # 对序列特征进行池化（取均值）
        pooled_output = torch.mean(sequence_output, dim=1).squeeze().cpu().numpy()

        return pooled_output

    def save_features_to_pkl(self, features, dataset_folder):
        """
        将蛋白质特征保存为 pkl 文件。
        """
        output_file = os.path.join(self.output_dir, dataset_folder, 'protein_features.pkl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 使用 torch.save 保存特征
        torch.save(features, output_file)

        print(f"Features for {dataset_folder} saved to {output_file}")

if __name__ == '__main__':
    # 设置数据目录和输出目录
    data_dir = '../data'  # 你的数据集根目录
    output_dir = '../data'  # 保存pkl文件的目录

    print("Starting feature extraction...")  # 添加调试输出

    # 创建并运行特征提取器
    feature_extractor = ProteinFeatureExtractor(data_dir=data_dir, output_dir=output_dir)
    feature_extractor.process_datasets()

    print("Feature extraction completed.")  # 添加调试输出
