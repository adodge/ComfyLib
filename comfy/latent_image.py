from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from comfy.hazard.utils import common_upscale
from comfy.util import SDType, _check_divisible_by_8


class UpscaleMethod(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    AREA = "area"


class CropMethod(Enum):
    DISABLED = "disabled"
    CENTER = "center"


class RGBImage(SDType):
    def __init__(self, data: Tensor, device: Union[str, torch.device] = "cpu"):
        self._data = data  # shape: (1, height, width, 3)
        self.to(device)

    def to(self, device: Union[str, torch.device]) -> "RGBImage":
        """
        Modifies the object in-place.
        """
        torch_device = torch.device(device)
        if torch_device == self.device:
            return self

        self._data = self._data.to(torch_device)
        self.device = torch_device
        return self

    def size(self) -> Tuple[int, int]:
        _, height, width, _ = self._data.size()
        return width, height

    def to_image(self) -> Image:
        arr = self._data.detach().cpu().numpy().reshape(self._data.shape[1:])
        arr = (np.clip(arr, 0, 1) * 255).round().astype("uint8")
        return Image.fromarray(arr)

    @classmethod
    def from_image(
        cls, image: Image, device: Union[str, torch.device] = "cpu"
    ) -> "RGBImage":
        img_a = np.array(image)
        assert img_a.ndim == 3
        height, width, channels = img_a.shape
        assert channels == 3
        img_t = Tensor(img_a.reshape((1, height, width, 3))/255)
        return cls(img_t, device=device)

    def to_tensor(self) -> Tensor:
        return self._data


class GreyscaleImage(SDType):
    def __init__(self, data: Tensor, device: Union[str, torch.device] = "cpu"):
        self._data = data  # shape: (height, width)
        self.to(device)

    def to(self, device: Union[str, torch.device]) -> "GreyscaleImage":
        """
        Modifies the object in-place.
        """
        torch_device = torch.device(device)
        if torch_device == self.device:
            return self

        self._data = self._data.to(torch_device)
        self.device = torch_device
        return self

    def size(self) -> Tuple[int, int]:
        height, width = self._data.size()
        return width, height

    def to_image(self) -> Image:
        arr = self._data.detach().cpu().numpy()
        arr = (np.clip(arr, 0, 1) * 255).round().astype("uint8")
        return Image.fromarray(arr)

    @classmethod
    def from_image(
        cls, image: Image, device: Union[str, torch.device] = "cpu"
    ) -> "GreyscaleImage":
        img_a = np.array(image)
        if img_a.ndim == 3:
            assert img_a.shape[2] == 1
            img_a = img_a.reshape(img_a.shape[2:])
        height, width = img_a.shape
        img_t = Tensor(img_a.reshape((height, width))/255)
        return cls(img_t, device=device)

    def to_tensor(self) -> Tensor:
        return self._data


class LatentImage(SDType):
    def __init__(
        self,
        data: Tensor,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self._data = data
        self._noise_mask: Optional[Tensor] = mask
        self.to(device)

    def to(self, device: Union[str, torch.device]) -> "LatentImage":
        """
        Modifies the object in-place.
        """
        torch_device = torch.device(device)
        if torch_device == self.device:
            return self

        self._data = self._data.to(torch_device)
        if self._noise_mask is not None:
            self._noise_mask = self._noise_mask.to(torch_device)
        self.device = torch_device
        return self

    def size(self) -> Tuple[int, int]:
        _, _, height, width = self._data.size()
        return width, height

    @classmethod
    def empty(cls, width: int, height: int, device: Union[str, torch.device] = "cpu"):
        # EmptyLatentImage
        width, height = _check_divisible_by_8(width, height)
        img = torch.zeros([1, 4, height, width])
        return cls(img, device=device)

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
            return LatentImage(s, latent_to._noise_mask, device=latent_to.device)

        s_from = latent_from._data[:, :, : height - y, : width - x]
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

        return LatentImage(s, latent_to._noise_mask, device=latent_to.device)

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
        return LatentImage(img, device=self.device)

    def set_mask(self, mask: GreyscaleImage) -> "LatentImage":
        # SetLatentNoiseMask
        assert mask.size() == self.size()
        return LatentImage(self._data, mask=mask.to_tensor(), device=self.device)

    def to_internal_representation(self):
        out = {"samples": self._data}
        if self._noise_mask is not None:
            out["noise_mask"] = self._noise_mask
        return out

    def to_arrays(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img: np.ndarray = self._data.detach().cpu().numpy()
        img = img.reshape(img.shape[1:])
        img = np.moveaxis(img, (0,1,2), (2,0,1))

        msk = None
        if self._noise_mask is not None:
            msk = self._noise_mask.detach().cpu().numpy()
            msk = np.moveaxis(msk, (0,1), (1,0))
        return img, msk

    @classmethod
    def from_arrays(
        cls, img_a: np.ndarray, mask_a: Optional[np.ndarray], device: Union[str, torch.device] = "cpu"
    ) -> "LatentImage":

        assert img_a.ndim == 3
        height, width, channels = img_a.shape
        assert channels == 4
        img_a = np.moveaxis(img_a, (2,0,1), (0,1,2))
        img_a = img_a.reshape((1, *img_a.shape))
        img_t = Tensor(img_a)

        mask_t = None
        if mask_a is not None:
            if mask_a.ndim == 3:
                assert mask_a.shape[2] == 1
                mask_a = mask_a.reshape(mask_a[:2])
            mask_a = np.moveaxis(mask_a, (1,0), (0,1))
            mask_t = Tensor(mask_a)



        return cls(img_t, mask=mask_t, device=device)