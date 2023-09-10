import torch


OBJ0 = {
    "y": 4.5,
    "x": 6.25,
    "h": 3.5,
    "w": 2.5,
    "cls": 2
}

OBJ1 = {
    "y": 1.75,
    "x": 2.,
    "h": 3.25,
    "w": 2.,
    "cls": 0
}


def get_obj_samples():
    batch_size, num_objects, d6 = (2, 5, 6)
    hw = 8
    num_classes = 3
    objects = torch.zeros((batch_size, num_objects, d6))

    # y, x, hy, wx, cls, confidence
    objects[0, 0, :] = torch.tensor([OBJ0['y'], OBJ0['x'], OBJ0['h'], OBJ0['w'], OBJ0['cls'], 1.])
    objects[0, 1, :] = torch.tensor([OBJ1['y'], OBJ1['x'], OBJ1['h'], OBJ1['w'], OBJ1['cls'], 1.])
    return objects
