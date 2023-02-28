from enum import Enum
from typing import Optional, Tuple

import torch
from PIL import Image
from torch import Tensor

from comfy.hazard.utils import common_upscale
from comfy.util import _check_divisible_by_8, _image_to_greyscale_tensor, DeviceLocal, Device, _get_torch_device, \
    _update_device


class UpscaleMethod(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    AREA = "area"


class CropMethod(Enum):
    DISABLED = "disabled"
    CENTER = "center"


class LatentImage(DeviceLocal):
    def __init__(self, data: Tensor, mask: Optional[Tensor] = None):
        self._data = data
        self._noise_mask: Optional[Tensor] = mask

    def to(self, device: Device):
        dev = _get_torch_device(device)
        self._data = self._data.to(dev)
        if self._noise_mask is not None:
            self._noise_mask = self._noise_mask.to(dev)

    def device(self) -> Device:
        out = None
        out = _update_device(out, self._data)
        if self._noise_mask is not None:
            out = _update_device(out, self._noise_mask)
        return out or Device.CPU

    def size(self) -> Tuple[int, int]:
        _, _, height, width = self._data.size()
        return width, height

    @classmethod
    def empty(cls, width: int, height: int):
        # EmptyLatentImage
        width, height = _check_divisible_by_8(width, height)
        img = torch.zeros([1, 4, height, width])
        return cls(img)

    @classmethod
    def combine(
        cls,
        latent_to: "LatentImage",
        latent_from: "LatentImage",
        x: int,
        y: int,
        feather: int,
    ) -> "LatentImage":
        # LatentComposite
        x, y, feather = _check_divisible_by_8(x, y, feather)

        assert latent_to.size() == latent_from.size()

        s = latent_to._data.clone()
        width, height = latent_from.size()

        if feather == 0:
            s[:, :, y : y + height, x : x + width] = latent_from._data[
                :, :, : height - y, : width - x
            ]
            return LatentImage(s, latent_to._noise_mask)

        s_from = latent_to._data[:, :, : height - y, : width - x]
        mask = torch.ones_like(s_from)

        for t in range(feather):
            c = (1.0 / feather) * (t + 1)
            if y != 0:
                mask[:, :, t : 1 + t, :] *= c
            if y + height < height:
                mask[:, :, height - 1 - t : height - t, :] *= c
            if x != 0:
                mask[:, :, :, t : 1 + t] *= c
            if x + width < width:
                mask[:, :, :, width - 1 - t : width - t] *= c

        rev_mask = torch.ones_like(mask) - mask
        s[:, :, y : y + height, x : x + width] = (
            s_from[:, :, : height - y, : width - x] * mask
            + s[:, :, y : y + height, x : x + width] * rev_mask
        )

        return LatentImage(s, latent_to._noise_mask)

    def upscale(
        self,
        width: int,
        height: int,
        upscale_method: UpscaleMethod,
        crop_method: CropMethod,
    ) -> "LatentImage":
        # LatentUpscale
        width, height = _check_divisible_by_8(width, height)

        img = common_upscale(
            self._data.clone().detach(),
            width,
            height,
            upscale_method.value,
            crop_method.value,
        )
        return LatentImage(img)

    def set_mask(self, mask: Image) -> "LatentImage":
        # SetLatentNoiseMask
        mask_t, mask_size = _image_to_greyscale_tensor(mask)
        assert mask_size == self.size()
        return LatentImage(self._data, mask_t)

    def to_internal_representation(self):
        out = {"samples": self._data}
        if self._noise_mask is not None:
            out["noise_mask"] = self._noise_mask
        return out
