""" Здесь находится 'Голова' CenterNet, описана в разделе 4 статьи https://arxiv.org/pdf/1904.07850.pdf"""
from torch import nn
import torch


class CenterNetHead(nn.Module):
    """
    Принимает на вход тензор из Backbone input[B, K, W/R, H/R], где
    - B = batch_size
    - K = количество каналов (в ResnetBackbone K = 64)
    - H, W = размеры изображения на вход Backbone
    - R = output stride, т.е. во сколько раз featuremap меньше, чем исходное изображение
      (в ResnetBackbone R = 4)

    Возвращает тензора [B, C+4, W/R, H/R]:
    - первые C каналов: probs[B, С, W/R, H/R] - вероятности от 0 до 1
    - еще 2 канала: offset[B, 2, W/R, H/R] - поправки координат в пикселях от 0 до 1
    - еще 2 канала: sizes[B, 2, W/R, H/R] - размеры объекта в пикселях
    """
    def __init__(self, k_in_channels=64, c_classes: int = 2):
        super().__init__()
        self.c_classes = c_classes
        raise NotImplementedError()


    def forward(self, input_t: torch.Tensor):
        raise NotImplementedError()
        return torch.cat([class_heatmap, offset_map, size_map], dim=1)
