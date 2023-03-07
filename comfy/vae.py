from typing import Union

import torch
from torch import Tensor

from comfy.hazard.sd import VAE
from comfy.latent_image import LatentImage, RGBImage, GreyscaleImage
from comfy.util import (
    _check_divisible_by_64, SDType,
)


class VAEModel(SDType):
    def __init__(self, model: VAE, device: Union[str, torch.device] = "cpu"):
        self._model = model
        self.to(device)

    def to(self, device: Union[str, torch.device]) -> "VAEModel":
        """
        Modifies the object in-place.
        """
        torch_device = torch.device(device)
        if torch_device == self.device:
            return self

        self._model.first_stage_model.to(torch_device)
        self._model.device = torch_device
        self.device = torch_device
        return self

    @classmethod
    def from_model(cls, model_filepath: str) -> "VAEModel":
        # VAELoader
        return VAEModel(VAE(ckpt_path=model_filepath))

    @SDType.requires_cuda
    def encode(self, image: RGBImage) -> LatentImage:
        # VAEEncode
        # XXX something's wrong here, I think\
        img = self._model.encode(image.to_tensor().to(self.device))
        return LatentImage(img, device=self.device)

    @SDType.requires_cuda
    def masked_encode(self, image: RGBImage, mask: GreyscaleImage) -> LatentImage:
        # VAEEncodeForInpaint

        image_t = image.to_tensor().clone()
        mask_t = image.to_tensor()

        assert image.size() == mask.size()
        _check_divisible_by_64(*image.size())

        kernel_tensor = torch.ones((1, 1, 6, 6))

        mask_erosion = torch.clamp(
            torch.nn.functional.conv2d(
                (1.0 - mask_t.round())[None], kernel_tensor, padding=3
            ),
            0,
            1,
        )

        for i in range(3):
            image_t[:, :, :, i] -= 0.5
            image_t[:, :, :, i] *= mask_erosion[0][:, :].round()
            image_t[:, :, :, i] += 0.5

        img = self._model.encode(image_t)
        return LatentImage(img, mask=mask_t, device=self.device)

    @SDType.requires_cuda
    def decode(self, latent_image: LatentImage) -> RGBImage:
        # VAEDecode

        img: Tensor = self._model.decode(
            latent_image.to_internal_representation()["samples"]
        )
        if img.shape[0] != 1:
            raise RuntimeError(
                f"Expected the output of vae.decode to have shape[0]==1.  shape={img.shape}"
            )
        return RGBImage(img, device=self.device)