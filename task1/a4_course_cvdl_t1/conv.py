import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(kernel_size % 2 == 1)
        super().__init__()
        raise NotImplementedError()

    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(tensor, one_side_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        pass

    @staticmethod
    def _cross_correlate(input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        raise NotImplementedError()
