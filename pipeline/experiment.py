import os
import random
import torch
import datetime
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 用于学习率衰减
from tqdm import tqdm  # 用于进度条

from pipeline.trainer import Trainer  # 导入 Trainer 类
from pipeline.evaluator import Evaluator  # 导入 Evaluator 类


class DTAExperiment:
    def __init__(self, config, model, train_loader, val_loader=None, test_loader=None,
                 cold_protein_loader=None, cold_drug_loader=None, cold_all_loader=None):
        """
        初始化 DTAExperiment 类。

        :param config: 配置参数，包括学习率、批大小等
        :param model: 要训练的模型
        :param train_loader: 训练数据的 DataLoader
        :param val_loader: 验证数据的 DataLoader（可选）
        :param test_loader: 测试数据的 DataLoader（可选）
        :param cold_protein_loader: 冷启动蛋白质验证集的 DataLoader（可选）
        :param cold_drug_loader: 冷启动药物验证集的 DataLoader（可选）
        :param cold_all_loader: 冷启动全数据验证集的 DataLoader（可选）
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join(config.output_dir, config.name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = model.to(self.device)

        if self.device.type == "cuda":
            logical_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(logical_id)
            print(f"✅ Model loaded on device: cuda:{logical_id} ({gpu_name})")
        else:
            print("✅ Model loaded on device: CPU")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=config.lr_scheduler_patience)

        self.trainer = Trainer(self.model, self.optimizer, device=self.device)
        self.evaluator = Evaluator(self.model, device=self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cold_protein_loader = cold_protein_loader
        self.cold_drug_loader = cold_drug_loader
        self.cold_all_loader = cold_all_loader

        # 使用硬编码的模型参数
        self.model_save_paths = {
            'val': os.path.join(self.output_dir, "best_model_val.pth"),
            'test': os.path.join(self.output_dir, "best_model_test.pth")
        }

        self.best_mae_dict = {'val': float("inf")}
        self.early_stop_counter = 0
        self.early_stop_best_mae = float("inf")
        self.early_stopping_patience = config.early_stopping_patience

        self.epoch_csv_path = os.path.join(self.output_dir, f"{config.name}_epoch_metrics_{self.timestamp}.csv")
        self.final_csv_path = os.path.join(self.output_dir, f"{config.name}_final_metrics_{self.timestamp}.csv")

        print(f"\n🚀 Starting new experiment!")
        print(f"   📂 Dataset: {self.config.dataset}")
        print(f"   💾 Output Directory: {self.output_dir}")
        print(f"   🔧 Total Epochs: {self.config.epochs}")

    def train(self):
        with open(self.epoch_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Test Set", "MSE", "RMSE", "MAE", "R2", "PEARSON", "SPEARMAN", "CI", "RM2"])

        experiment_iter = tqdm(range(1, self.config.epochs + 1), desc="Training Progress", ncols=100)

        for epoch in experiment_iter:
            train_loss = self.trainer.train_epoch(self.train_loader)
            print(f"\n🎯 Epoch {epoch}: 🔧 Train Loss: {train_loss:.4f}")

            # 根据数据集名称判断是否是冷启动模式
            if "cold_start" in self.config.dataset:
                self._evaluate_cold_start(epoch)
            else:
                self._evaluate_standard(epoch)

            # EarlyStopping逻辑 (按MAE)
            self._early_stopping(epoch)

    def _evaluate_cold_start(self, epoch):
        # 仅在冷启动模式下验证，不再进行测试集验证
        for name, loader in [
            ('cold_protein', self.cold_protein_loader),
            ('cold_drug', self.cold_drug_loader),
            ('cold_all', self.cold_all_loader)
        ]:
            if loader is not None:
                val_metrics = self.evaluator.evaluate(loader)
                self._log_metrics(epoch, name, val_metrics)

                # 保存最佳模型
                val_mae = val_metrics["MAE"]
                if val_mae < self.best_mae_dict.get(name, float("inf")):
                    self.best_mae_dict[name] = val_mae
                    torch.save(self.model.state_dict(), self.model_save_paths[name])
                    print(f"✅ [{name}] Best model saved at epoch {epoch} with MAE={val_mae:.4f}")

    def _evaluate_standard(self, epoch):
        # 仅在标准数据集上进行评估，并保存最佳模型
        val_metrics = self.evaluator.evaluate(self.val_loader)
        self._log_metrics(epoch, "val", val_metrics)

        # 保存最佳模型
        val_mae = val_metrics["MAE"]
        if val_mae < self.best_mae_dict.get('val', float("inf")):
            self.best_mae_dict['val'] = val_mae
            torch.save(self.model.state_dict(), self.model_save_paths['val'])  # 保存在 val 目录
            print(f"✅ [val] Best model saved at epoch {epoch} with MAE={val_mae:.4f}")

    def _log_metrics(self, epoch, test_name, val_metrics):
        with open(self.epoch_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [epoch, test_name] + [f"{val_metrics[k]:.4f}" for k in ["MSE", "RMSE", "MAE", "R2", "PEARSON", "SPEARMAN", "CI", "RM2"]]
            writer.writerow(row)

    def _early_stopping(self, epoch):
        # EarlyStopping逻辑 (按MAE)
        monitored_mae = self.best_mae_dict.get('val', float("inf"))

        if monitored_mae < self.early_stop_best_mae:
            self.early_stop_best_mae = monitored_mae
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        if self.early_stop_counter >= self.early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs without improvement.")
            return

        self.scheduler.step(monitored_mae)

    def evaluate_final(self):
        if not self.test_loader:
            print("⚠ No test set provided, skipping final evaluation.")
            return

        with open(self.final_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Test Set", "MSE", "RMSE", "MAE", "R2", "PEARSON", "SPEARMAN", "CI", "RM2"]
            writer.writerow(header)

            # 加载最佳模型
            print(f"\n📥 Loading best model for final evaluation ...")
            self.model.load_state_dict(torch.load(self.model_save_paths['val']))  # 加载验证集的最佳模型

            # 在测试集上进行评估
            metrics = self.evaluator.evaluate(self.test_loader)
            print(f"📌 Final Test Metrics (test):")
            for k, v in metrics.items():
                print(f"   {k:>10}: {v:.4f}")

            writer.writerow(["test"] + [f"{metrics[k]:.4f}" for k in header[1:]])
