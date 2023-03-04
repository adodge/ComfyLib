from PIL import Image

import comfy.latent_image
import comfy.stable_diffusion

config = comfy.stable_diffusion.CheckpointConfig.from_built_in(comfy.stable_diffusion.BuiltInCheckpointConfigName.V1)

sd, clip, vae = comfy.stable_diffusion.load_checkpoint(
    config=config,
    checkpoint_filepath="../stable-diffusion-webui/models/Stable-diffusion/sd_base/v1-5-pruned-emaonly.safetensors",
    embedding_directory=None,
)

pos = clip.encode("an astronaut")
neg = clip.encode("")

image0 = Image.open("input.png")
latentA = vae.encode(image0)

latentB = sd.sample(positive=pos, negative=neg, latent_image=latentA, seed=42, steps=20, cfg_scale=7,
                    sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER, scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=0.75)


image1 = vae.decode(latentB)
image1.save("out1.png")

latentC = latentB.upscale(768, 768, comfy.latent_image.UpscaleMethod.NEAREST, comfy.latent_image.CropMethod.DISABLED)

latentD = sd.sample(positive=pos, negative=neg, latent_image=latentC, seed=42, steps=20, cfg_scale=7,
                    sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER, scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=0.75)

image2 = vae.decode(latentC)
image2.save("out2.png")

image3 = vae.decode(latentD)
image3.save("out3.png")
