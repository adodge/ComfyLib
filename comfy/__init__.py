import io
import os.path
from enum import Enum
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import Tensor

from comfy.hazard.nodes import common_ksampler
from comfy.hazard.sd import load_checkpoint as _load_checkpoint, ModelPatcher, VAE, CLIP
from comfy.hazard.utils import common_upscale

def _divide_size_by_n(width: int, height: int, n: int) -> Tuple[int, int]:
    if width % n != 0 or height % n != 0:
        raise ValueError(f"width and height must be multiples of {n} (actual: {width}, {height})")
    return width // n, height // n


def _divide_size_by_64(width: int, height: int) -> Tuple[int, int]:
    return _divide_size_by_n(width, height, 64)


def _divide_size_by_8(width: int, height: int) -> Tuple[int, int]:
    return _divide_size_by_n(width, height, 8)


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
    def __init__(self, data: Tensor):
        self._data = data

    @classmethod
    def empty(cls, width: int, height: int):
        width, height = _divide_size_by_8(width, height)
        img = torch.zeros([1, 4, height, width])
        return cls(img)

    def latent_upscale(
            self, width: int, height: int, upscale_method: UpscaleMethod, crop_method: CropMethod
    ) -> "LatentImage":
        width, height = _divide_size_by_8(width, height)

        img = common_upscale(self._data.clone().detach(), width, height, upscale_method.value, crop_method.value)
        return LatentImage(img)

    def to_internal_representation(self):
        return {"samples": self._data}


class Conditioning:
    def __init__(self, data: Tensor):
        self._data = data

    def to_internal_representation(self):
        return [[self._data, {}]]


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
    def from_checkpoint(cls, checkpoint_filepath: str, config: CheckpointConfig):
        stable_diffusion, _, _ = _load_checkpoint(
            config_path=config.to_file_like(),
            ckpt_path=checkpoint_filepath,
            output_vae=False, output_clip=False)
        return cls(stable_diffusion)

    def sample(
            self, positive: Conditioning, negative: Conditioning, latent_image: LatentImage,
            seed: int, steps: int, cfg_scale: float, sampler: Sampler, scheduler: Scheduler,
            denoise_strength: float,
    ) -> LatentImage:
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

        return LatentImage(img[0]['samples'])


class VAEModel:
    def __init__(self, model: VAE):
        self._model = model

    def encode(self, image: Image) -> LatentImage:
        arr = np.array(image)
        height, width, channels = arr.shape
        assert channels == 3
        _divide_size_by_64(width, height)
        arr = Tensor(arr.reshape((1, height, width, channels)))
        img = self._model.encode(arr)
        return LatentImage(img)

    def decode(self, latent_image: LatentImage) -> Image:
        img: Tensor = self._model.decode(latent_image.to_internal_representation()['samples'])
        if img.shape[0] != 1:
            raise RuntimeError(f"Expected the output of vae.decode to have shape[0]==1.  shape={img.shape}")
        arr = img.detach().cpu().numpy().reshape(img.shape[1:])
        arr = (np.clip(arr, 0, 1) * 255).round().astype("uint8")
        return Image.fromarray(arr)


class CLIPModel:
    def __init__(self, model: CLIP):
        self._model = model

    def encode(self, text: str) -> Conditioning:
        result = self._model.encode(text)
        return Conditioning(result)


def load_checkpoint(
        checkpoint_filepath: str, config: CheckpointConfig, embedding_directory: Optional[str]
) -> Tuple[StableDiffusionModel, CLIPModel, VAEModel]:
    stable_diffusion, clip, vae = _load_checkpoint(
        config.to_file_like(), checkpoint_filepath,
        output_vae=True, output_clip=True,
        embedding_directory=embedding_directory)

    return StableDiffusionModel(stable_diffusion), CLIPModel(clip), VAEModel(vae)
