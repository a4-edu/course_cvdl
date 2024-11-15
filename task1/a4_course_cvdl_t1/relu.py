import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Принимает x, возвращает y(x)
        """
        raise NotImplementedError()

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Принимат dL/dy, возвращает dL/dx.
        """
        raise NotImplementedError()

