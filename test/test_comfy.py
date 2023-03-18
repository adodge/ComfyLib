import logging
import os
from unittest import TestCase

import numpy as np
import pytest
import torch.cuda
import gc

from PIL import Image
from torch import Tensor

import comfy.stable_diffusion
import comfy.clip
import comfy.vae
import comfy.conditioning
import comfy.latent_image
from comfy.util import ModelLoadError

V1_CHECKPOINT_FILEPATH = os.environ.get("V1_CHECKPOINT_FILEPATH")
V1_SAFETENSORS_FILEPATH = os.environ.get("V1_SAFETENSORS_FILEPATH")
V2_SAFETENSORS_FILEPATH = os.environ.get("V2_SAFETENSORS_FILEPATH")


class TestImageConversions(TestCase):
    def test_rgb_image_roundtrip(self):
        r1: np.ndarray = (np.clip(np.random.random((10,20, 3)), 0, 1) * 255).round().astype("uint8")
        img1 = Image.fromarray(r1)
        rgb = comfy.latent_image.RGBImage.from_image(img1)
        img2 = rgb.to_image()
        r2 = np.array(img2)
        self.assertTrue(np.all(r1 == r2))

    def test_greyscale_image_roundtrip(self):
        r1: np.ndarray = (np.clip(np.random.random((10,20)), 0, 1) * 255).round().astype("uint8")
        img1 = Image.fromarray(r1)
        grey = comfy.latent_image.GreyscaleImage.from_image(img1)
        img2 = grey.to_image()
        r2 = np.array(img2)
        self.assertEqual(r1.shape, r2.shape)
        self.assertTrue(np.all(r1 == r2))

    def test_latent_to_array_roundtrip(self):

        r1: np.ndarray = np.arange(64*64*4).reshape((64,64,4))
        latent1 = comfy.latent_image.LatentImage.from_arrays(r1, None)
        r2, _ = latent1.to_arrays()

        self.assertEqual(r1.shape, r2.shape)
        self.assertTrue(np.all(r1 == r2))


class TestSDV1(TestCase):
    @classmethod
    def setUpClass(cls):
        name = comfy.stable_diffusion.BuiltInCheckpointConfigName.V1
        config = comfy.stable_diffusion.CheckpointConfig.from_built_in(name)
        cls.sd, cls.clip, cls.vae = comfy.stable_diffusion.load_checkpoint(V1_CHECKPOINT_FILEPATH, config)

    def setUp(self) -> None:
        self.sd.to("cpu")
        self.clip.to("cpu")
        self.vae.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_max_memory_allocated()

    def tearDown(self) -> None:
        logging.info(f"Test used max {torch.cuda.max_memory_allocated()}")

    def test_load_checkpoint(self):
        self.assertIsInstance(self.sd, comfy.stable_diffusion.StableDiffusionModel)
        self.assertIsInstance(self.clip, comfy.clip.CLIPModel)
        self.assertIsInstance(self.vae, comfy.vae.VAEModel)
        self.assertEqual(self.sd.version, comfy.stable_diffusion.SDVersion.SD1x)

    @torch.no_grad()
    def test_text_to_image(self):
        latent = comfy.latent_image.LatentImage.empty(512, 512, device="cuda")

        self.clip.to("cuda")
        pos = self.clip.encode("An astronaut")
        neg = self.clip.encode("bad hands")
        self.clip.to("cpu")

        self.sd.to("cuda")
        result = self.sd.sample(positive=pos, negative=neg, latent_image=latent, seed=0, steps=1, cfg_scale=7,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER,
                                scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=1.0)
        self.sd.to("cpu")

        self.vae.to("cuda")
        image = self.vae.decode(result)
        self.vae.to("cpu")

        image_pillow = image.to_image()

        image_pillow.save("text2image.png")

    @torch.no_grad()
    def test_image_to_image(self):

        input_image = Image.open("example.png")
        input_image_t = comfy.latent_image.RGBImage.from_image(input_image, device="cuda")

        self.vae.to("cuda")
        latent = self.vae.encode(input_image_t)
        self.vae.to("cpu")

        self.clip.to("cuda")
        pos = self.clip.encode("a woman with wings")
        neg = self.clip.encode("watermark, text")
        self.clip.to("cpu")

        self.sd.to("cuda")
        result = self.sd.sample(positive=pos, negative=neg, latent_image=latent, seed=0, steps=2, cfg_scale=8,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_DPMpp_2M,
                                scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=0.8)
        self.sd.to("cpu")

        self.vae.to("cuda")
        image = self.vae.decode(result)
        self.vae.to("cpu")

        image_pillow = image.to_image()

        image_pillow.save("image2image.png")

    @torch.no_grad()
    def test_hires_fix(self):
        latent = comfy.latent_image.LatentImage.empty(768, 768, device="cuda")

        self.clip.to("cuda")
        pos = self.clip.encode("An astronaut")
        neg = self.clip.encode("bad hands")
        self.clip.to("cpu")

        self.sd.to("cuda")
        result = self.sd.sample(positive=pos, negative=neg, latent_image=latent, seed=0, steps=2, cfg_scale=8,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_DPMpp_SDE,
                                scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=1.0)
        self.sd.to("cpu")

        result2 = result.upscale(1152, 1152, upscale_method=comfy.latent_image.UpscaleMethod.NEAREST,
                       crop_method=comfy.latent_image.CropMethod.DISABLED)

        self.sd.to("cuda")
        result3 = self.sd.sample(positive=pos, negative=neg, latent_image=result2, seed=0, steps=2, cfg_scale=8,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_DPMpp_SDE,
                                scheduler=comfy.stable_diffusion.Scheduler.SIMPLE, denoise_strength=0.5)
        self.sd.to("cpu")

        self.vae.to("cuda")
        image = self.vae.decode(result3)
        self.vae.to("cpu")

        image_pillow = image.to_image()

        image_pillow.save("hiresfix.png")

    @torch.no_grad()
    def test_V1_model_V2_config_errors(self):
        name = comfy.stable_diffusion.BuiltInCheckpointConfigName.V2
        config = comfy.stable_diffusion.CheckpointConfig.from_built_in(name)

        with self.assertRaises(ModelLoadError):
            comfy.stable_diffusion.load_checkpoint(V1_CHECKPOINT_FILEPATH, config)

    @torch.no_grad()
    def test_clip_version_propagation(self):
        cond = self.clip.encode("Foo")
        self.assertEqual(cond.version, comfy.conditioning.ConditioningVersion.SD1x)


class TestSDV2(TestCase):
    @classmethod
    def setUpClass(cls):
        name = comfy.stable_diffusion.BuiltInCheckpointConfigName.V2
        config = comfy.stable_diffusion.CheckpointConfig.from_built_in(name)
        cls.sd, cls.clip, cls.vae = comfy.stable_diffusion.load_checkpoint(V2_SAFETENSORS_FILEPATH, config)

    def setUp(self) -> None:
        self.sd.to("cpu")
        self.clip.to("cpu")
        self.vae.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_max_memory_allocated()

    def tearDown(self) -> None:
        logging.info(f"Test used max {torch.cuda.max_memory_allocated()}")

    def test_load_checkpoint(self):
        self.assertIsInstance(self.sd, comfy.stable_diffusion.StableDiffusionModel)
        self.assertIsInstance(self.clip, comfy.clip.CLIPModel)
        self.assertIsInstance(self.vae, comfy.vae.VAEModel)
        self.assertEqual(self.sd.version, comfy.stable_diffusion.SDVersion.SD2x)

    @torch.no_grad()
    def test_text_to_image(self):
        latent = comfy.latent_image.LatentImage.empty(768, 768, device="cuda")

        self.clip.to("cuda")
        pos = self.clip.encode("An astronaut")
        neg = self.clip.encode("bad hands")
        self.clip.to("cpu")

        self.sd.to("cuda")
        result = self.sd.sample(positive=pos, negative=neg, latent_image=latent, seed=0, steps=20, cfg_scale=7,
                                sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER,
                                scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=1.0)
        self.sd.to("cpu")

        self.vae.to("cuda")
        image = self.vae.decode(result)
        self.vae.to("cpu")

        image_pillow = image.to_image()

        image_pillow.save("text2image_v2.png")

    @torch.no_grad()
    def test_V2_model_V1_config_errors(self):
        name = comfy.stable_diffusion.BuiltInCheckpointConfigName.V1
        config = comfy.stable_diffusion.CheckpointConfig.from_built_in(name)

        with self.assertRaises(ModelLoadError):
            comfy.stable_diffusion.load_checkpoint(V2_SAFETENSORS_FILEPATH, config)

    @torch.no_grad()
    def test_clip_version_propagation(self):
        cond = self.clip.encode("Foo")
        self.assertEqual(cond.version, comfy.conditioning.ConditioningVersion.SD2x)


class TestGuessConfig(TestCase):
    def test_v1(self):
        sd, clip, vae = comfy.stable_diffusion.load_checkpoint(V1_SAFETENSORS_FILEPATH)
        self.assertIsInstance(sd, comfy.stable_diffusion.StableDiffusionModel)
        self.assertIsInstance(clip, comfy.clip.CLIPModel)
        self.assertIsInstance(vae, comfy.vae.VAEModel)
        self.assertEqual(sd.version, comfy.stable_diffusion.SDVersion.SD1x)

    def test_v2(self):
        sd, clip, vae = comfy.stable_diffusion.load_checkpoint(V2_SAFETENSORS_FILEPATH)
        self.assertIsInstance(sd, comfy.stable_diffusion.StableDiffusionModel)
        self.assertIsInstance(clip, comfy.clip.CLIPModel)
        self.assertIsInstance(vae, comfy.vae.VAEModel)
        self.assertEqual(sd.version, comfy.stable_diffusion.SDVersion.SD2x)