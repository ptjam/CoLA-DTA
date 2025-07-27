import pandas as pd
import numpy as np
import random
import os
from sklearn.utils import shuffle

def split_dataset(path, output_dir='standard', train_ratio=0.8, val_ratio=0.1, seed=114514):
    random.seed(seed)
    np.random.seed(seed)

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据，只保留三列
    data = pd.read_csv(path, sep='\t')[['compound_id', 'PDB_ID', 'label']]
    data = shuffle(data, random_state=seed).reset_index(drop=True)

    all_size = len(data)
    train_size = int(all_size * train_ratio)
    val_size = int(all_size * val_ratio)
    test_size = all_size - train_size - val_size

    train_set = []
    val_set = []
    test_set = []

    compound_in_train = set()
    protein_in_train = set()

    for idx, row in data.iterrows():
        compound = row['compound_id']
        protein = row['PDB_ID']  

        # 先填充训练集
        if len(train_set) < train_size:
            train_set.append(row)
            compound_in_train.add(compound)
            protein_in_train.add(protein)
        else:
            # 尝试放入验证集和测试集
            if len(val_set) < val_size:
                if compound in compound_in_train or protein in protein_in_train:
                    val_set.append(row)
                else:
                    train_set.append(row)
                    compound_in_train.add(compound)
                    protein_in_train.add(protein)
            else:
                if compound in compound_in_train or protein in protein_in_train:
                    test_set.append(row)
                else:
                    train_set.append(row)
                    compound_in_train.add(compound)
                    protein_in_train.add(protein)

    # 最后转为DataFrame
    train_df = pd.DataFrame(train_set)
    val_df = pd.DataFrame(val_set)
    test_df = pd.DataFrame(test_set)

    print(f"最终数据量: 训练集={len(train_df)}, 验证集={len(val_df)}, 测试集={len(test_df)}")

    # 保存结果
    train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    tsv_file = 'data/PDB/all.tsv'  # 替换为你的数据集路径
    output_dir = 'data/PDB/standard'  # 输出目录
    split_dataset(path=tsv_file, output_dir=output_dir, train_ratio=0.8, val_ratio=0.1, seed=114514)
