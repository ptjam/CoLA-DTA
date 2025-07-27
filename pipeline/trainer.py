import torch
import torch.nn as nn

from utils.device_utils import move_to_device

class Trainer:
    def __init__(self, model, optimizer, loss_fn=None, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = move_to_device(batch, self.device)
            labels = batch['label'].view(-1)
            preds = self.model(batch).view(-1)
            loss = self.loss_fn(preds, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
