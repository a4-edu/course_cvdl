import torch

from abbyy_course_cvdl_t2.network import CenterNet, PointsNonMaxSuppression
from abbyy_course_cvdl_t2.convert import ObjectsToPoints, PointsToObjects

from .obj_samples import OBJ0, OBJ1, get_obj_samples


def test_nms():
    batch_size, num_objects, d6 = (2, 5, 6)
    hw = 8
    num_classes = 3
    objects = get_obj_samples()
    objects[0, 0] *= 0
    assert objects.shape == (batch_size, num_objects, d6)

    kernel_size = 3

    obj_to_points = ObjectsToPoints(hw=hw, num_classes=num_classes, smooth_kernel_size=kernel_size)
    points_to_objs = PointsToObjects(objects_per_image=16)
    points_without_nms = obj_to_points(objects)
    objects_without_nms = points_to_objs(points_without_nms)

    assert torch.allclose(objects_without_nms[1], torch.zeros_like(objects_without_nms[1])), objects_without_nms[1]
    objects_without_nms = objects_without_nms[0][objects_without_nms[0][:, -1] > 0.]
    assert len(objects_without_nms) == kernel_size * kernel_size, objects_without_nms

    nms = PointsNonMaxSuppression(kernel_size=kernel_size)
    points_with_nms = nms(points_without_nms)
    objects_with_nms = points_to_objs(points_with_nms)

    assert torch.allclose(objects_with_nms[1], torch.zeros_like(objects_with_nms[1])), objects_with_nms[1]
    objects_with_nms = objects_with_nms[0][objects_with_nms[0][:, -1] > 0.]
    assert len(objects_with_nms) == 1, objects_with_nms


def test_network():
    net = CenterNet()