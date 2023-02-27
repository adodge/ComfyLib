import numpy as np
import torch
from PIL import Image
from torch import Tensor

from comfy import LatentImage
from comfy.hazard.sd import VAE
from comfy.util import (
    _check_divisible_by_64,
    _image_to_greyscale_tensor,
    _image_to_rgb_tensor,
)


class VAEModel:
    def __init__(self, model: VAE):
        self._model = model

    @classmethod
    def from_model(cls, model_filepath: str) -> "VAEModel":
        # VAELoader
        return VAEModel(VAE(ckpt_path=model_filepath))

    def encode(self, image: Image) -> LatentImage:
        # VAEEncode
        img_t, _ = _image_to_rgb_tensor(image)
        img = self._model.encode(img_t)
        return LatentImage(img)

    def masked_encode(self, image: Image, mask: Image) -> LatentImage:
        # VAEEncodeForInpaint
        img_t, img_size = _image_to_rgb_tensor(image)
        mask_t, mask_size = _image_to_greyscale_tensor(mask)
        assert img_size == mask_size
        _check_divisible_by_64(*img_size)

        kernel_tensor = torch.ones((1, 1, 6, 6))

        mask_erosion = torch.clamp(
            torch.nn.functional.conv2d(
                (1.0 - mask_t.round())[None], kernel_tensor, padding=3
            ),
            0,
            1,
        )

        for i in range(3):
            img_t[:, :, :, i] -= 0.5
            img_t[:, :, :, i] *= mask_erosion[0][:, :].round()
            img_t[:, :, :, i] += 0.5

        img = self._model.encode(img_t)
        return LatentImage(img, mask=mask_t)

    def decode(self, latent_image: LatentImage) -> Image:
        # VAEDecode

        img: Tensor = self._model.decode(
            latent_image.to_internal_representation()["samples"]
        )
        if img.shape[0] != 1:
            raise RuntimeError(
                f"Expected the output of vae.decode to have shape[0]==1.  shape={img.shape}"
            )
        arr = img.detach().cpu().numpy().reshape(img.shape[1:])
        arr = (np.clip(arr, 0, 1) * 255).round().astype("uint8")
        return Image.fromarray(arr)
