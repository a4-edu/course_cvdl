import numpy as np
import torch
from torch import nn
from torch.utils import data as tdata

from tqdm import tqdm


class Trainer:
    def __init__(self, *,
        net: nn.Module, opt: torch.optim.Optimizer,
        train_loader: tdata.DataLoader, val_loader: tdata.DataLoader, test_loader: tdata.DataLoader,
        loss: nn.Module, metric: nn.Module,
    ):
        self.net = net
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.loss = loss
        self.metric = metric

        self.device = list(net.parameters())[0].device

    def run_epoch(self, loader, log_freq=10):
        is_training = self.net.training

        loss_values = []
        metric_values = []

        for num, (x, y) in tqdm(enumerate((iter(loader)))):
            x, y = x.to(self.device), y.to(self.device)
            yp = self.net(x)
            loss_val = self.loss(yp, y).mean()
            metric_val = self.metric(yp, y).mean()

            if is_training:
                self.opt.zero_grad()
                loss_val.backward()
                self.opt.step()

            if num % log_freq == 0:
                loss_values.append(loss_val.detach().cpu().item())
                metric_values.append(metric_val.detach().cpu().numpy())

        print(
            f"Stats: Loss={np.mean(loss_values):.2f} Metric={np.mean(metric_values):.2f}"
        )
        return loss_values, metric_values

    def __call__(self, num_epochs, with_testing=True):
        stats = {"train": [], "val": [], "test": []}
        for e in range(num_epochs):
            print(f"Training:")
            self.net.train()
            epoch_stats = self.run_epoch(self.train_loader)
            stats["train"].append(epoch_stats)

            print(f"Validating:")
            self.net.eval()
            epoch_stats = self.run_epoch(self.val_loader, log_freq=1)
            stats["val"].append(epoch_stats)

        if with_testing:
            self.net.eval()
            print(f"Testing:")
            epoch_stats = self.run_epoch(self.test_loader, log_freq=1)
            stats["test"].append(epoch_stats)
        return stats
