from typing import Union

import torch
from torch import Tensor

from .hazard.sd import VAE
from .latent_image import GreyscaleImage, LatentImage, RGBImage
from .util import SDType, _check_divisible_by_64, _check_divisible_by_8


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
    def from_model(
        cls, model_filepath: str, device: Union[str, torch.device] = "cpu"
    ) -> "VAEModel":
        # VAELoader
        return VAEModel(VAE(ckpt_path=model_filepath), device=device)

    def encode(self, image: RGBImage) -> LatentImage:
        # VAEEncode
        # XXX something's wrong here, I think
        _check_divisible_by_64(*image.size())
        img = self._model.encode(image.to_tensor().to(self.device))
        return LatentImage(img, device=self.device)

    def masked_encode(self, image: RGBImage, mask: GreyscaleImage) -> LatentImage:
        # VAEEncodeForInpaint

        image_t = image.to_tensor().clone()
        mask_t = mask.to_tensor()

        assert image.size() == mask.size()
        _check_divisible_by_64(*image.size())

        kernel_tensor = torch.ones((1, 1, 6, 6)).to(self.device)

        mask_erosion = torch.clamp(
            torch.nn.functional.conv2d((mask_t.round()[None, None]), kernel_tensor, padding=3),
            0, 1)

        m = 1.0-mask_t.round()

        for i in range(3):
            image_t[:, :, :, i] -= 0.5
            image_t[:, :, :, i] *= m
            image_t[:, :, :, i] += 0.5

        img = self._model.encode(image_t)
        return LatentImage(img, mask=mask_erosion[0].round(), device=self.device)

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
