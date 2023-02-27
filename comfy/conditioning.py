from typing import Dict, Iterable, List, Optional, Tuple

from torch import Tensor

from .util import _check_divisible_by_8


class Conditioning:
    def __init__(self, data: Optional[Tensor], meta: Optional[Dict] = None):
        meta = meta or {}
        self._data: List[Tuple[Tensor, Dict]] = []
        if data is not None:
            self._data.append((data, meta))

    @classmethod
    def combine(cls, inputs: Iterable["Conditioning"]) -> "Conditioning":
        # ConditioningCombine
        out = cls(None)
        for cond in inputs:
            out._data.extend(cond._data)
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

        c = Conditioning(None)

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
