import os
import random
import torch
from argparse import Namespace
from torch_geometric.loader import DataLoader

from models.dta_model import DTARegressionModel
from models.dta_model_no_cross_attention import DTARegressionModelNoCrossAttention
from models.dta_model_no_fusion import DTARegressionModelNoFusion
from models.dta_model_no_protein_graph import DTARegressionModelNoProteinGraph
from models.dta_model_no_compound_graph import DTARegressionModelWithoutCompoundGraph
from dataset.cpidataset import CPIDataset, collate_fn
from pipeline.experiment import DTAExperiment


# ==================== 工具函数 ====================

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Seed set to {seed}\n")


# ==================== 硬编码数据集定义 ====================

datasets = [
    {"name": "DAVIS"},
    {"name": "KIBA"},
    {"name": "BDB"},
    {"name": "PDB"},
    {"name": "PDB_cold_start"}
]

seeds = [42]


# ==================== 加载数据集和DataLoader ====================

def load_dataloader(dataset_name, batch_size=8):
    """
    根据给定的数据集名称，加载对应的 DataLoader，并返回一个或多个 DataLoader。
    :param dataset_name: 数据集名称（如 'DAVIS', 'KIBA', 'BDB', 'PDB', 'PDB_cold_start'）
    :param batch_size: 批次大小
    """
    # 硬编码路径
    file_paths = {
        "DAVIS": {
            "train": 'data/DAVIS/standard/train.tsv',
            "val": 'data/DAVIS/standard/val.tsv',
            "test": 'data/DAVIS/standard/test.tsv',
            "protein_graph": 'data/DAVIS/protein_graph',
            "compound_graph": 'data/DAVIS/compound_graphs',
            "protein_feat": 'data/DAVIS/protein_esm_feature.pkl',
            "compound_feat": 'data/DAVIS/compound_bert_features.pkl'
        },
        "KIBA": {
            "train": 'data/KIBA/standard/train.tsv',
            "val": 'data/KIBA/standard/val.tsv',
            "test": 'data/KIBA/standard/test.tsv',
            "protein_graph": 'data/KIBA/protein_graph',
            "compound_graph": 'data/KIBA/compound_graphs',
            "protein_feat": 'data/KIBA/protein_esm_feature.pkl',
            "compound_feat": 'data/KIBA/compound_bert_features.pkl'
        },
        "BDB": {
            "train": 'data/BDB/standard/train.tsv',
            "val": 'data/BDB/standard/val.tsv',
            "test": 'data/BDB/standard/test.tsv',
            "protein_graph": 'data/BDB/protein_graph',
            "compound_graph": 'data/BDB/compound_graphs',
            "protein_feat": 'data/BDB/protein_esm_feature.pkl',
            "compound_feat": 'data/BDB/compound_bert_features.pkl'
        },
        "PDB": {
            "train": 'data/PDB/standard/train.tsv',
            "val": 'data/PDB/standard/val.tsv',
            "test": 'data/PDB/standard/val.tsv',
            "protein_graph": 'data/PDB/protein_graph',
            "compound_graph": 'data/PDB/compound_graphs',
            "protein_feat": 'data/PDB/protein_esm_feature.pkl',
            "compound_feat": 'data/PDB/compound_bert_features.pkl'
        },
        "PDB_cold_start": {
            "train": 'data/PDB/cold_start/train.tsv',
            "cold_protein": 'data/PDB/cold_start/cold_protein.tsv',
            "cold_drug": 'data/PDB/cold_start/cold_drug.tsv',
            "cold_all": 'data/PDB/cold_start/cold_all.tsv',
            "protein_graph": 'data/PDB/protein_graph',
            "compound_graph": 'data/PDB/compound_graphs',
            "protein_feat": 'data/PDB/protein_esm_feature.pkl',
            "compound_feat": 'data/PDB/compound_bert_features.pkl'
        }
    }

    if dataset_name not in file_paths:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    paths = file_paths[dataset_name]

    if dataset_name == "PDB_cold_start":
        # 返回四个 cold-start DataLoader
        train_dataset = CPIDataset(paths["train"], paths["protein_graph"], paths["compound_graph"],
                                   paths["protein_feat"], paths["compound_feat"], preload=True)
        cold_protein_dataset = CPIDataset(paths["cold_protein"], paths["protein_graph"], paths["compound_graph"],
                                          paths["protein_feat"], paths["compound_feat"], preload=True)
        cold_drug_dataset = CPIDataset(paths["cold_drug"], paths["protein_graph"], paths["compound_graph"],
                                       paths["protein_feat"], paths["compound_feat"], preload=True)
        cold_all_dataset = CPIDataset(paths["cold_all"], paths["protein_graph"], paths["compound_graph"],
                                      paths["protein_feat"], paths["compound_feat"], preload=True)

        print(f"🔄 Preloading {dataset_name} dataset (cold-start) ...")
        print(f"✅ {dataset_name} - train dataset size: {len(train_dataset)} samples")
        print(f"✅ {dataset_name} - cold_protein dataset size: {len(cold_protein_dataset)} samples")
        print(f"✅ {dataset_name} - cold_drug dataset size: {len(cold_drug_dataset)} samples")
        print(f"✅ {dataset_name} - cold_all dataset size: {len(cold_all_dataset)} samples")

        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn),
            "cold_protein": DataLoader(cold_protein_dataset, batch_size=batch_size, collate_fn=collate_fn),
            "cold_drug": DataLoader(cold_drug_dataset, batch_size=batch_size, collate_fn=collate_fn),
            "cold_all": DataLoader(cold_all_dataset, batch_size=batch_size, collate_fn=collate_fn)
        }

    # 标准模式：train, val, test
    train_dataset = CPIDataset(paths["train"], paths["protein_graph"], paths["compound_graph"], paths["protein_feat"],
                               paths["compound_feat"], preload=True)
    val_dataset = CPIDataset(paths["val"], paths["protein_graph"], paths["compound_graph"], paths["protein_feat"],
                             paths["compound_feat"], preload=True)
    test_dataset = CPIDataset(paths["test"], paths["protein_graph"], paths["compound_graph"], paths["protein_feat"],
                              paths["compound_feat"], preload=True)

    print(f"🔄 Preloading {dataset_name} dataset (standard) ...")
    print(f"✅ {dataset_name} - train dataset size: {len(train_dataset)} samples")
    print(f"✅ {dataset_name} - val dataset size: {len(val_dataset)} samples")
    print(f"✅ {dataset_name} - test dataset size: {len(test_dataset)} samples")

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn),
        "val": DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn),
        "test": DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    }


# ==================== 统一训练逻辑 ====================

def run_experiment(config, dataset_name, seed, model, dataloaders):
    print(f"\n🚀 ===== Running experiment for {dataset_name} with seed {seed} =====")

    # 设置随机种子
    set_seed(seed)

    train_cfg = config['training']
    batch_size = train_cfg['batch_size']
    epochs = train_cfg['epochs']
    lr = train_cfg['lr']
    weight_decay = train_cfg['weight_decay']
    lr_scheduler_patience = train_cfg.get('lr_scheduler_patience', 20)
    early_stopping_patience = train_cfg.get('early_stopping_patience', 30)

    # 获取模型的名称，并根据此修改保存路径
    model_class_name = model.__class__.__name__
    experiment_name = f"{dataset_name}_{model_class_name}_seed{seed}"

    # 动态生成模型的输出路径
    model_output_dir = os.path.join(config['output']['output_dir'], model_class_name, experiment_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # 更新配置
    experiment_config = Namespace(
        seed=seed,
        name=experiment_name,
        dataset=dataset_name,
        model_class=model.__class__,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"] if dataset_name != "PDB_cold_start" else None,
        test_loader=dataloaders["test"] if dataset_name != "PDB_cold_start" else None,
        cold_protein_loader=dataloaders["cold_protein"] if dataset_name == "PDB_cold_start" else None,
        cold_drug_loader=dataloaders["cold_drug"] if dataset_name == "PDB_cold_start" else None,
        cold_all_loader=dataloaders["cold_all"] if dataset_name == "PDB_cold_start" else None,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        lr_scheduler_patience=lr_scheduler_patience,
        early_stopping_patience=early_stopping_patience,
        collate_fn=collate_fn,
        output_dir=model_output_dir,  # 使用新的模型输出目录
        resume_path=None
    )

    # 创建并运行实验
    if (dataset_name != 'PDB_cold_start'):
        experiment = DTAExperiment(experiment_config, model, train_loader=dataloaders["train"],
                                   val_loader=dataloaders["val"], test_loader=dataloaders["test"])
        experiment.train()
        experiment.evaluate_final()
    else:
        experiment = DTAExperiment(experiment_config, model, train_loader=dataloaders["train"],
                                   cold_all_loader=dataloaders["cold_all"],
                                   cold_protein_loader=dataloaders["cold_protein"],
                                   cold_drug_loader=dataloaders["cold_drug"])
        experiment.train()
        experiment.evaluate_final()


# ==================== 主控入口 ====================

if __name__ == '__main__':
    # 配置硬编码
    config = {
        'training': {
            'batch_size': 256,
            'epochs': 200,
            'lr': 0.001,
            'weight_decay': 0.00001,
            'lr_scheduler': 'plateau',
            'lr_scheduler_patience': 10,
            'early_stopping_patience': 20
        },
        'output': {
            'output_dir': 'saved_models/ablation_study',
            'model_dir': 'checkpoints/ablation_study',
            'log_dir': 'logs',
            'save_best_only': True
        }
    }

    # 你可以在这里导入多个模型，如：
    models = [
        DTARegressionModel,
        DTARegressionModelNoCrossAttention,
        DTARegressionModelNoFusion,
        DTARegressionModelNoProteinGraph,
        DTARegressionModelWithoutCompoundGraph,
    ]

    # 遍历数据集并执行实验
    for ds in datasets:
        for seed in seeds:
            print(f"\n🚀 ===== Running experiment for {ds['name']} with seed {seed} =====")

            # 获取当前数据集的名称
            dataset_name = ds['name']

            # 加载数据集和DataLoader
            dataloaders = load_dataloader(dataset_name, batch_size=config['training']['batch_size'])

            # 针对每个模型，逐个执行训练和评估
            for model in models:
                # 直接实例化模型（无需传递任何超参数）
                model_instance = model()  # 直接调用，无需传递参数
                # 执行实验
                run_experiment(config, dataset_name, seed, model_instance, dataloaders)

    print("\n🎉 All experiments finished successfully.\n")
