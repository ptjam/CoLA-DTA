import os

def get_dataset_paths(name: str, mode: str):
    """
    根据数据集名称与模式返回路径集合，兼容 standard 与 cold-start 模式。

    参数：
        name (str): 数据集名称，如 PDB / DAVIS / KIBA / BDB
        mode (str): 模式，standard 或 cold-start

    返回：
        dict: 路径字典
    """
    name = name.upper()
    base_dir = os.path.join("data", name)

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"数据目录不存在: {base_dir}")

    paths = {
        # 通用图与特征路径（两种模式都需要）
        "protein_graph_dir": os.path.join(base_dir, "protein_graph"),
        "compound_graph_dir": os.path.join(base_dir, "compound_graphs"),
        "protein_feat_pkl": os.path.join(base_dir, "protein_esm_feature.pkl"),
        "compound_feat_pkl": os.path.join(base_dir, "compound_bert_features.pkl"),
    }

    if mode == "standard":
        standard_dir = os.path.join(base_dir, "standard")
        paths.update({
            "train_tsv": os.path.join(standard_dir, "train.tsv"),
            "val_tsv": os.path.join(standard_dir, "val.tsv"),
            "test_tsv": os.path.join(standard_dir, "test.tsv"),  # 注意：PDB可能没有test.tsv，后续在加载时判断
        })
    elif mode == "cold-start":
        split_dir = os.path.join(base_dir, "cold_start")
        paths.update({
            "train_tsv": os.path.join(split_dir, "train.tsv"),
            "cold_all_tsv": os.path.join(split_dir, "cold_all.tsv"),
            "cold_protein_tsv": os.path.join(split_dir, "cold_protein.tsv"),
            "cold_drug_tsv": os.path.join(split_dir, "cold_drug.tsv"),
        })
    else:
        raise ValueError(f"未知模式: {mode}，请使用 'standard' 或 'cold-start'")

    return paths
