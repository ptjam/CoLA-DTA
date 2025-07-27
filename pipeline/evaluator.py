import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils.device_utils import move_to_device

class Evaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = move_to_device(batch, self.device)
                pred = self.model(batch).view(-1).cpu().numpy()
                label = batch['label'].view(-1).cpu().numpy()
                preds.extend(pred)
                labels.extend(label)

        preds = np.array(preds)
        labels = np.array(labels)
        return self.compute_metrics(preds, labels)

    def compute_metrics(self, preds, labels):
        mse = np.mean((preds - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - labels))
        var_y = np.var(labels)
        r2 = 1 - mse / var_y if var_y != 0 else 0.0

        # 处理pearson与spearman的特殊情况，避免nan
        if len(preds) < 2:
            pearson = 0.0
            spearman = 0.0
        else:
            pearson = pearsonr(preds, labels)[0]
            spearman = spearmanr(preds, labels)[0]

        ci = self.concordance_index(preds, labels)
        rm2 = self.rm2_score(preds, labels)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "PEARSON": pearson,
            "SPEARMAN": spearman,
            "CI": ci,
            "RM2": rm2
        }

    def concordance_index(self, preds, labels):
        n = 0
        h_sum = 0.0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if labels[i] != labels[j]:
                    n += 1
                    if (preds[i] < preds[j] and labels[i] < labels[j]) or (preds[i] > preds[j] and labels[i] > labels[j]):
                        h_sum += 1
                    elif preds[i] == preds[j]:
                        h_sum += 0.5
        return h_sum / n if n != 0 else 0.0

    def rm2_score(self, preds, labels):
        if len(preds) < 2:
            return 0.0
        r = pearsonr(preds, labels)[0]
        r2 = r ** 2
        return r2 * (1 - (np.sqrt(r2) - r) ** 2)
