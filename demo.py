from PIL import Image

import comfy

config = comfy.CheckpointConfig.from_built_in(comfy.BuiltInCheckpointConfigName.V1)

sd, clip, vae = comfy.load_checkpoint(
    config=config,
    checkpoint_filepath="../stable-diffusion-webui/models/Stable-diffusion/sd_base/v1-5-pruned-emaonly.safetensors",
    embedding_directory=None,
)

pos = clip.encode("an astronaut")
neg = clip.encode("")

image0 = Image.open("input.png")
latentA = vae.encode(image0)
latentB = sd.sample(positive=pos, negative=neg, latent_image=latentA, seed=42, steps=20, cfg_scale=7,
                 sampler=comfy.Sampler.SAMPLE_EULER, scheduler=comfy.Scheduler.NORMAL, denoise_strength=0.75)
image1 = vae.decode(latentB)
image1.save("out1.png")

latent0 = comfy.LatentImage.empty(512, 512)
latent1 = sd.sample(positive=pos, negative=neg, latent_image=latent0, seed=42, steps=20, cfg_scale=7,
                 sampler=comfy.Sampler.SAMPLE_EULER, scheduler=comfy.Scheduler.NORMAL, denoise_strength=1.0)
image = vae.decode(latent_image=latent1)

image.save("out2.png")