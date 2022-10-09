import segmentation_models_pytorch as smp
from torch import nn

from .data import SuimDataset


class SuimModel(nn.Module):
    def __init__(self, freeze_encoder:bool=True):
        super().__init__()
        self.net = smp.Linknet(
            encoder_weights="imagenet",
            classes=len(SuimDataset.LABEL_COLORS),
            activation="softmax",
        )
        if freeze_encoder:
            for name, param in self.net.named_parameters():
                if ("encoder" in name) and ("conv_stem" not in name):
                    param.requires_grad = False

    def forward(self, x):
        return self.net(x)
