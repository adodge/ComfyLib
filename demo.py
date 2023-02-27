from PIL import Image

import comfy
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

latent0 = comfy.latent_image.LatentImage.empty(512, 512)
latent1 = sd.sample(positive=pos, negative=neg, latent_image=latent0, seed=42, steps=20, cfg_scale=7,
                    sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER, scheduler=comfy.stable_diffusion.Scheduler.NORMAL, denoise_strength=1.0)
image = vae.decode(latent_image=latent1)

image.save("out2.png")