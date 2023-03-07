import functools
from typing import Iterable, Optional

import torch


def _check_divisible_by_64(*vals: int) -> Iterable[int]:
    return _check_divisible_by_n(64, *vals)


def _check_divisible_by_8(*vals: int) -> Iterable[int]:
    return _check_divisible_by_n(8, *vals)


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