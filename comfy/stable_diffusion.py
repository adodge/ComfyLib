import io
import os
from enum import Enum
from typing import Optional, Tuple

from omegaconf import OmegaConf

from comfy import Conditioning, LatentImage
from comfy.clip import CLIPModel
from comfy.hazard.nodes import common_ksampler
from comfy.hazard.sd import ModelPatcher
from comfy.hazard.sd import load_checkpoint as _load_checkpoint
from comfy.vae import VAEModel


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
