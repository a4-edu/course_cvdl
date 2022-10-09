import torch
from torch import nn


class DiceLoss(torch.nn.Module):
    def forward(self, predict, target):
        """
        Для каждого класса:
            DICE[C] = 1 - 2TP / ( 2TP + FP + FN )
        Результат: Mean(DICE)
        """
        if predict.shape != target.shape:
            b, num_classes, h, w = predict.shape
            target = torch.nn.functional.one_hot(target.long(), num_classes=num_classes)
            target = target.squeeze(1).permute(0, 3, 1, 2)

        axis = [2, 3]
        tp = (predict * target).sum(axis=axis)
        fp = (predict * (1 - target)).sum(axis=axis)
        fn = ((1 - predict) * target).sum(axis=axis)

        eps = 1e-3
        perclass_index = 2 * tp / (2 * tp + fp + fn + eps)
        loss = 1 - perclass_index
        return loss.mean(axis=1)
