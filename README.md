(WIP)

This is the [ComfyUI](https://github.com/comfyanonymous/ComfyUI), but without
the UI.  It's stripped down and packaged as a library, for use in other projects.

ComfyUI is actively maintained (as of writing), and has implementations of a
lot of the cool cutting-edge Stable Diffusion stuff.  The implementation is as
unmodified as possible, to hopefully make it easier to merge in upstream
improvements.  In order to provide a consistent API, an interface layer has
been added.  Directly importing names not in the API should be considered
dangerous, as the ComfyUI maintainers might change how things are arranged
whenever they want.  The interface layer will be consistent though, within a
major version of the library.

# Installation

For now you can install from github:

```
pip3 install git+https://github.com/adodge/Comfy-Lib
```

# Example

```python3
import comfy

# Read in a checkpoint
sd, clip, vae = comfy.load_checkpoint(
    config_filepath="comfy/configs/v1-inference.yaml",
    checkpoint_filepath="v1-5-pruned-emaonly.safetensors",
    embedding_directory=None,
)

# CLIP encode a prompt
pos = clip.encode("an astronaut")
neg = clip.encode("")

# Run the sampler
img0 = comfy.LatentImage.empty(512, 512)
img1 = sd.sample(positive= pos, negative=neg, latent_image=img0, seed=42, steps=20, cfg_strength=7,
                 sampler=comfy.Sampler.SAMPLE_EULER, scheduler=comfy.Scheduler.NORMAL, denoise_strength=1.0)

# Run the VAE to get a Pillow Image
img2 = vae.decode(latent_image=img1)

# Save that to a file
img2.save("out.png")
```

# API

## (WIP)

