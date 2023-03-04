import functools
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def _check_divisible_by_64(*vals: int) -> Iterable[int]:
    return _check_divisible_by_n(64, *vals)


def _check_divisible_by_8(*vals: int) -> Iterable[int]:
    return _check_divisible_by_n(8, *vals)


def _image_to_rgb_tensor(image: Image) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Tensor with shape [1, height, width, 3]
    """
    imgA = np.array(image)
    assert imgA.ndim == 3
    height, width, channels = imgA.shape
    assert channels == 3
    imgT = Tensor(imgA.reshape((1, height, width, 3)))
    return imgT, (width, height)


def _image_to_greyscale_tensor(image: Image) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Tensor with shape [height, width]
    """
    imgA = np.array(image)
    if imgA.ndim == 3:
        assert imgA.shape[2] == 1
        imgA = imgA.reshape(imgA.shape[2:])
    height, width = imgA.shape
    imgT = Tensor(imgA.reshape((height, width)))
    return imgT, (width, height)


def _check_divisible_by_n(n: int, *vals: int) -> Iterable[int]:
    for v in vals:
        if v % n != 0:
            raise ValueError(f"Expected an integer divisible by {n}, but got {v}")
    return (v // n for v in vals)

class SDType:
    device: Optional[torch.device] = None

    @classmethod
    def requires_cuda(cls, func):
        @functools.wraps(func)
        def f(self, *args, **kwargs):
            if self.device is None:
                raise RuntimeError
            if not self.device.type == "cuda":
                raise RuntimeError(f"{cls.__name__} must be moved to CUDA before calling {func.__name__}")
            return func(self, *args, **kwargs)
        return f