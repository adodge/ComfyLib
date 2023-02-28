import abc
from enum import Enum
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


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MIXED = "mixed"


def _get_torch_device(device: Device) -> torch.device:
    if device == Device.CPU:
        return torch.device("cpu")
    elif device == Device.CUDA:
        return torch.device("cuda")
    else:
        raise ValueError(f"Invalid device to move to: {device}")


def _update_device(device: Optional[Device], obj: Tensor) -> Device:
    if obj.device.type == "cpu":
        object_device = Device.CPU
    elif obj.device.type == "cuda":
        object_device = Device.CUDA
    else:
        raise RuntimeError(f"object has an unexpected device type: {obj.device}")
    if device is None or device == object_device:
        return object_device
    else:
        return Device.MIXED


class DeviceLocal:#(abc.ABC):
    """
    A class for objects that can be moved from one device to another.
    """
    #@abc.abstractmethod
    def to(self, device: Device):
        raise NotImplementedError

    #@abc.abstractmethod
    @property
    def device(self) -> Device:
        raise NotImplementedError
