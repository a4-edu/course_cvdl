import torch
from abbyy_course_cvdl_t2.loss import CenterNetLoss, ObjectsToPoints

from .obj_samples import OBJ0, OBJ1, get_obj_samples


def test_loss():
    batch_size, num_objects, d6 = (2, 5, 6)
    hw = 8
    num_classes = 3
    obj_to_points = ObjectsToPoints(hw=hw, num_classes=num_classes, smooth_kernel_size=0)
    crit = CenterNetLoss(obj_to_points=obj_to_points, l_size_lambda=1, r_scale=1)

    objects = torch.zeros((batch_size, num_objects, d6))
    objects[0, 0, :] = torch.tensor([OBJ0['y'], OBJ0['x'], OBJ0['h'], OBJ0['w'], OBJ0['cls'], 1.])
    objects[0, 1, :] = torch.tensor([OBJ1['y'], OBJ1['x'], OBJ1['h'], OBJ1['w'], OBJ1['cls'], 1.])
    points = obj_to_points(objects)

    loss = crit(points, objects)
    assert loss.shape == (batch_size, 3), loss.shape
    loss_lk, loss_offset, loss_sizes = torch.unbind(loss, axis=-1)
    assert loss_lk.shape == (batch_size,), loss_lk.shape
    assert (loss_lk >=0).all(), loss_lk
    assert loss_offset.shape == (batch_size,), loss_offset.shape
    assert (loss_offset >=0).all(), loss_offset
    assert loss_sizes.shape == (batch_size,), loss_sizes.shape
    assert (loss_sizes >=0).all(), loss_offset
