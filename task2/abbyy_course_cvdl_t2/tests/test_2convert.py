from abbyy_course_cvdl_t2.convert import ObjectsToPoints, PointsToObjects
from math import floor
import torch
import numpy as np

from .obj_samples import OBJ0, OBJ1, get_obj_samples


def test_objects_to_points():
    batch_size, num_objects, d6 = (2, 5, 6)
    hw = 8
    num_classes = 3
    objects = get_obj_samples()
    assert objects.shape == (batch_size, num_objects, d6)
    # objects = torch.zeros((batch_size, num_objects, d6))

    # # y, x, hy, wx, cls, confidence
    # objects[0, 0, :] = torch.tensor([OBJ0['y'], OBJ0['x'], OBJ0['h'], OBJ0['w'], OBJ0['cls'], 1.])
    # objects[0, 1, :] = torch.tensor([OBJ1['y'], OBJ1['x'], OBJ1['h'], OBJ1['w'], OBJ1['cls'], 1.])

    idx0 = floor(OBJ0['y']), floor(OBJ0['x'])
    idx1 = floor(OBJ1['y']), floor(OBJ1['x'])

    obj_to_points = ObjectsToPoints(hw=hw, num_classes=num_classes, smooth_kernel_size=0)
    points = obj_to_points(objects)
    assert points.shape == (batch_size, num_classes + 4, hw, hw)
    probs, offsets, sizes = torch.split(points, [num_classes, 2, 2], dim=1)
    assert probs.sum() == 2, probs.sum()
    assert probs[1].sum() == 0, probs[0].sum()

    # class probs
    assert probs[0, OBJ0['cls'], idx0[0], idx0[1]] == 1.
    assert probs[0, OBJ1['cls'], idx1[0], idx1[1]] == 1.

    # Offsets
    assert offsets[1].sum() == 0
    # dy
    assert offsets[0, 0, idx0[0], idx0[1]] == (OBJ0['y'] - idx0[0])
    assert offsets[0, 0, idx1[0], idx1[1]] == (OBJ1['y'] - idx1[0])
    # dx
    assert offsets[0, 1, idx0[0], idx0[1]] == (OBJ0['x'] - idx0[1])
    assert offsets[0, 1, idx1[0], idx1[1]] == (OBJ1['x'] - idx1[1])

    # Sizes
    assert sizes[1].sum() == 0
    # hy
    assert sizes[0, 0, idx0[0], idx0[1]] == OBJ0['h']
    assert sizes[0, 0, idx1[0], idx1[1]] == OBJ1['h']
    # wx
    assert sizes[0, 1, idx0[0], idx0[1]] == OBJ0['w']
    assert sizes[0, 1, idx1[0], idx1[1]] == OBJ1['w']

    # check smooth works at least somehow
    obj_to_points = ObjectsToPoints(hw=hw, num_classes=num_classes, smooth_kernel_size=3)
    points2 = obj_to_points(objects)


def test_points_to_objects():
    batch_size, num_objects, d6 = (2, 5, 6)
    hw = 8
    num_classes = 3
    objects = torch.zeros((batch_size, num_objects, d6))

    # y, x, hy, wx, cls, confidence
    objects[0, 0, :] = torch.tensor([OBJ0['y'], OBJ0['x'], OBJ0['h'], OBJ0['w'], OBJ0['cls'], 1.])
    objects[0, 1, :] = torch.tensor([OBJ1['y'], OBJ1['x'], OBJ1['h'], OBJ1['w'], OBJ1['cls'], 1.])

    obj_to_points = ObjectsToPoints(hw=hw, num_classes=num_classes, smooth_kernel_size=0)
    points_to_objects = PointsToObjects(objects_per_image=num_objects)
    with torch.no_grad():
        points = obj_to_points(objects)
        objects_back = points_to_objects(points)

    assert objects.shape == objects_back.shape, objects_back.shape

    assert torch.allclose(objects[1], objects_back[1]), objects_back

    objects_0 = objects[0]
    objects_back_0 = objects_back[0]

    assert (objects_0[objects_0[:, -1] == 1.0]).shape == (objects_back_0[objects_back_0[:, -1] == 1.0]).shape
    objects_0 = objects_0[objects_0[:, -1] == 1.0].numpy()
    objects_back_0 = objects_back_0[objects_back_0[:, -1] == 1.0].numpy()

    are_objects_equal = np.allclose(objects_0, objects_back_0)
    are_objects_equal_but_reversed = np.allclose(objects_0, objects_back_0[::-1])
    assert (are_objects_equal or are_objects_equal_but_reversed), (objects_0, objects_back_0)