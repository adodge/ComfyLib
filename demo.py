import comfy

config = comfy.CheckpointConfig.from_built_in(comfy.BuiltInCheckpointConfigName.V1)

sd, clip, vae = comfy.load_checkpoint(
    config=config,
    checkpoint_filepath="../stable-diffusion-webui/models/Stable-diffusion/sd_base/v1-5-pruned-emaonly.safetensors",
    embedding_directory=None,
)

pos = clip.encode("an astronaut")
neg = clip.encode("")

img0 = comfy.LatentImage.empty(512, 512)
img1 = sd.sample(positive=pos, negative=neg, latent_image=img0, seed=42, steps=20, cfg_strength=7,
                 sampler=comfy.Sampler.SAMPLE_EULER, scheduler=comfy.Scheduler.NORMAL, denoise_strength=1.0)
img2 = vae.decode(latent_image=img1)

img2.save("out.png")