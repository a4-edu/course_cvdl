import torch

from abbyy_course_cvdl_t2.backbone import HeadlessPretrainedResnet18Encoder, HeadlessResnet18Encoder
from abbyy_course_cvdl_t2.backbone import UpscaleTwiceLayer, ResnetBackbone


def test_encoders():
    x = torch.rand((4, 3, 256, 256))
    encoder = HeadlessResnet18Encoder()
    y1 = encoder(x)
    assert y1.shape == (4, 512, 8, 8), y1.shape
    encoder_pretrained = HeadlessPretrainedResnet18Encoder()
    y2 = encoder_pretrained(x)
    assert y2.shape == (4, 512, 8, 8), y2.shape


def test_upscale_twice_layer():
    up1 = UpscaleTwiceLayer(16, 8)
    x = torch.rand((4, 16, 8, 8))
    y = up1(x)
    assert y.shape == (4, 8, 16, 16), y.shape


def test_backbone():
    out_channels = 63
    backbone = ResnetBackbone(out_channels=out_channels)
    x = torch.rand((4, 3, 256, 256))
    y = backbone(x)
    assert y.shape == ((4, 63, 64, 64)), y.shape
