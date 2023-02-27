import io
import os.path
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor

from comfy.hazard.nodes import common_ksampler
from comfy.hazard.sd import CLIP, VAE, ModelPatcher
from comfy.hazard.sd import load_checkpoint as _load_checkpoint
from comfy.hazard.sd import load_clip
from comfy.hazard.utils import common_upscale


def _check_divisible_by_n(n: int, *vals: int) -> Iterable[int]:
    for v in vals:
        if v % n != 0:
            raise ValueError(f"Expected an integer divisible by {n}, but got {v}")
    return (v // n for v in vals)


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


class Sampler(Enum):
    SAMPLE_EULER = "sample_euler"
    SAMPLE_EULER_ANCESTRAL = "sample_euler_ancestral"
    SAMPLE_HEUN = "sample_heun"
    SAMPLE_DPM_2 = "sample_dpm_2"
    SAMPLE_DPM_2_ANCESTRAL = "sample_dpm_2_ancestral"
    SAMPLE_LMS = "sample_lms"
    SAMPLE_DPM_FAST = "sample_dpm_fast"
    SAMPLE_DPM_ADAPTIVE = "sample_dpm_adaptive"
    SAMPLE_DPMpp_2S_ANCESTRAL = "sample_dpmpp_2s_ancestral"
    SAMPLE_DPMpp_SDE = "sample_dpmpp_sde"
    SAMPLE_DPMpp_2M = "sample_dpmpp_2m"
    DDIM = "ddim"
    UNI_PC = "uni_pc"
    UNI_PC_BH2 = "uni_pc_bh2"


class Scheduler(Enum):
    KARRAS = "karras"
    NORMAL = "normal"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"


class UpscaleMethod(Enum):
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    AREA = "area"


class CropMethod(Enum):
    DISABLED = "disabled"
    CENTER = "center"


class LatentImage:
    def __init__(self, data: Tensor, mask: Optional[Tensor] = None):
        self._data = data
        self._noise_mask: Optional[Tensor] = mask

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


class BuiltInCheckpointConfigName(Enum):
    V1 = "v1-inference.yaml"
    V2 = "v2-inference.yaml"


class CheckpointConfig:
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)

    @classmethod
    def from_built_in(cls, name: BuiltInCheckpointConfigName):
        path = os.path.join(os.path.dirname(__file__), "configs", name.value)
        return cls(config_path=path)

    def to_file_like(self):
        file_like = io.StringIO()
        OmegaConf.save(self.config, f=file_like)
        file_like.seek(0)
        return file_like


class StableDiffusionModel:
    def __init__(self, model: ModelPatcher):
        self._model = model

    @classmethod
    def from_checkpoint(
        cls, checkpoint_filepath: str, config: CheckpointConfig
    ) -> "StableDiffusionModel":
        # CheckpointLoader
        stable_diffusion, _, _ = _load_checkpoint(
            config_path=config.to_file_like(),
            ckpt_path=checkpoint_filepath,
            output_vae=False,
            output_clip=False,
        )
        return cls(stable_diffusion)

    def sample(
        self,
        positive: Conditioning,
        negative: Conditioning,
        latent_image: LatentImage,
        seed: int,
        steps: int,
        cfg_scale: float,
        sampler: Sampler,
        scheduler: Scheduler,
        denoise_strength: float,
    ) -> LatentImage:
        # KSampler

        device = "cuda"
        img = common_ksampler(
            device=device,
            model=self._model,
            seed=seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=sampler.value,
            scheduler=scheduler.value,
            positive=positive.to_internal_representation(),
            negative=negative.to_internal_representation(),
            latent=latent_image.to_internal_representation(),
            denoise=denoise_strength,
        )

        return LatentImage(img[0]["samples"])

    def advanced_sample(
        self,
        positive: Conditioning,
        negative: Conditioning,
        latent_image: LatentImage,
        seed: int,
        steps: int,
        cfg_scale: float,
        sampler: Sampler,
        scheduler: Scheduler,
        denoise_strength: float,
        start_at_step: int,
        end_at_step: int,
        add_noise: bool,
        return_with_leftover_noise: bool,
    ) -> LatentImage:
        # KSamplerAdvanced
        device = "cuda"

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        img = common_ksampler(
            device=device,
            model=self._model,
            seed=seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=sampler.value,
            scheduler=scheduler.value,
            positive=positive.to_internal_representation(),
            negative=negative.to_internal_representation(),
            latent=latent_image.to_internal_representation(),
            denoise=denoise_strength,
            force_full_denoise=force_full_denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
        )

        return LatentImage(img[0]["samples"])


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


class CLIPModel:
    def __init__(self, model: CLIP):
        self._model = model

    @classmethod
    def from_model(
        cls,
        model_filepath: str,
        stop_at_clip_layer: int = -1,
        embedding_directory: Optional[str] = None,
    ) -> "CLIPModel":
        # CLIPLoader

        clip = load_clip(
            ckpt_path=model_filepath, embedding_directory=embedding_directory
        )
        clip.clip_layer(stop_at_clip_layer)

        return CLIPModel(clip)

    def encode(self, text: str) -> Conditioning:
        # CLIPTextEncode
        result = self._model.encode(text)
        return Conditioning(result)


def load_checkpoint(
    checkpoint_filepath: str,
    config: CheckpointConfig,
    embedding_directory: Optional[str] = None,
) -> Tuple[StableDiffusionModel, CLIPModel, VAEModel]:
    # CheckpointLoader
    stable_diffusion, clip, vae = _load_checkpoint(
        config.to_file_like(),
        checkpoint_filepath,
        output_vae=True,
        output_clip=True,
        embedding_directory=embedding_directory,
    )

    return StableDiffusionModel(stable_diffusion), CLIPModel(clip), VAEModel(vae)
