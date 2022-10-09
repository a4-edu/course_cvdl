import torch
from torch import nn


class Accuracy(nn.Module):
    def forward(self, input_prob: torch.Tensor, target: torch.Tensor):
        input_cls = torch.argmax(input_prob, dim=1, keepdims=True)
        tp = (input_cls == target).byte().sum()
        total = target.numel()
        return tp / total
