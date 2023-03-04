import os
from unittest import TestCase

import torch.cuda
from PIL import Image
import gc

import comfy.stable_diffusion
import comfy.clip
import comfy.vae
import comfy.conditioning
import comfy.latent_image

V1_CHECKPOINT_FILEPATH = os.environ.get("V1_CHECKPOINT_FILEPATH")
V1_SAFETENSORS_FILEPATH = os.environ.get("V1_SAFETENSORS_FILEPATH")


class TestComfy(TestCase):
    @classmethod
    def setUpClass(cls):
        name = comfy.stable_diffusion.BuiltInCheckpointConfigName.V1
        config = comfy.stable_diffusion.CheckpointConfig.from_built_in(name)
        cls.sd, cls.clip, cls.vae = comfy.stable_diffusion.load_checkpoint(V1_CHECKPOINT_FILEPATH, config)

    def setUp(self):
        self.sd.to("cpu")
        self.clip.to("cpu")
        self.vae.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def test_load_checkpoint(self):
        self.assertIsInstance(self.sd, comfy.stable_diffusion.StableDiffusionModel)
        self.assertIsInstance(self.clip, comfy.clip.CLIPModel)
        self.assertIsInstance(self.vae, comfy.vae.VAEModel)

    def test_text_to_image(self):
        self.sd.to("cuda")
        self.clip.to("cuda")
        self.vae.to("cuda")

        latent = comfy.latent_image.LatentImage.empty(512, 512).to("cuda")

        pos = self.clip.encode("An astronaut")
        neg = self.clip.encode("bad hands")

        result = self.sd.sample(positive=pos, negative=neg, latent_image=latent, seed=0, steps=1, cfg_scale=7,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER,
                                scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=1.0)

        image = self.vae.decode(result)
