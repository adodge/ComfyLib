from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from comfy.util import SDType, _check_divisible_by_8

# Metadata keys: TODO class
# area, strength, min_sigma, max_sigma


class ConditioningVersion(Enum):
    SD1x = "SD1.x"
    SD2x = "SD2.x"


class Conditioning(SDType):
    def __init__(
        self,
        data: Optional[Tensor],
        meta: Optional[Dict] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        meta = meta or {}
        self._data: List[Tuple[Tensor, Dict]] = []
        if data is not None:
            self._data.append((data, meta))
            data.size()

        self.to(device)

        self.version: Optional[ConditioningVersion] = None
        for d in self._data:
            if d[0].size()[2] == 1024:
                self.version = ConditioningVersion.SD2x
            elif d[0].size()[2] == 768:
                self.version = ConditioningVersion.SD1x
            break


    def to(self, device: Union[str, torch.device]) -> "Conditioning":
        """
        Modifies the object in-place.
        """
        torch_device = torch.device(device)
        if torch_device == self.device:
            return self

        self._data = [(d.to(torch_device), m) for d, m in self._data]
        self.device = torch_device
        return self

    @classmethod
    def combine(cls, inputs: Iterable["Conditioning"]) -> "Conditioning":
        # ConditioningCombine
        out = cls(None)
        device = None
        for cond in inputs:
            out._data.extend(cond._data)
            if device is None:
                device = cond.device
        if device is not None:
            out.to(device)
        return out

    def set_area(
        self,
        width: int,
        height: int,
        x: int,
        y: int,
        strength: float,
        min_sigma: float = 0.0,
        max_sigma: float = 99.0,
    ) -> "Conditioning":
        # ConditioningSetArea
        width, height = _check_divisible_by_8(width, height)
        x, y = _check_divisible_by_8(x, y)

        c = Conditioning(None, device=self.device)

        for t, m in self._data:
            n = (t, m.copy())
            n[1]["area"] = (height, width, y, x)
            n[1]["strength"] = strength
            n[1]["min_sigma"] = min_sigma
            n[1]["max_sigma"] = max_sigma
            c._data.append(n)
        return c

    def to_internal_representation(self):
        return [[d, m] for d, m in self._data]
