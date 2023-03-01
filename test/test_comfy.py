import os
from unittest import TestCase

import torch.cuda

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
        torch.cuda.empty_cache()

    def test_load_checkpoint(self):
        self.assertIsInstance(self.sd, comfy.stable_diffusion.StableDiffusionModel)
        self.assertIsInstance(self.clip, comfy.clip.CLIPModel)
        self.assertIsInstance(self.vae, comfy.vae.VAEModel)
        self.assertEqual(0, torch.cuda.memory_allocated("cuda"))

    def test_text_to_image(self):
        pos = self.clip.encode("An astronaut")
        neg = self.clip.encode("bad hands")
        self.assertIsInstance(pos, comfy.conditioning.Conditioning)
        self.assertIsInstance(neg, comfy.conditioning.Conditioning)

        latent = comfy.latent_image.LatentImage.empty(512, 512)
        self.assertIsInstance(latent, comfy.latent_image.LatentImage)

        # So far none of this should be in VRAM
        self.assertEqual(0, torch.cuda.memory_allocated("cuda"))

        result = self.sd.sample(positive=pos, negative=neg, latent_image=latent, seed=0, steps=1, cfg_scale=7,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER,
                                scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=1.0)
        self.assertIsInstance(result, comfy.latent_image.LatentImage)

        # Something should be in VRAM now
        self.assertNotEqual(0, torch.cuda.memory_allocated("cuda"))

        print(self.sd._model.device)