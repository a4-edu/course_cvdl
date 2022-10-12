import torch
from abbyy_course_cvdl_t2.head import CenterNetHead


def test_head():
    k_in_channels = 64
    c_classes = 3

    head = CenterNetHead(k_in_channels=k_in_channels, c_classes=c_classes)
    x = torch.rand((4, k_in_channels, 64, 64))
    y = head(x)

    assert y.shape == (4, c_classes + 4, 64, 64), y.shape
    probs, offsets, sizes = torch.split(y,[c_classes, 2, 2], dim=1)

    assert (probs >= 0).all(), probs
    assert (probs <= 1).all()

