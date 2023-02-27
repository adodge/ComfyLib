# This is a work in progress

All the talk about having a reliable interface below is aspirational.  The
interface is in flux at the moment, and no guarantee of the master branch
working are made, yet.  Soon.

----

This is the [ComfyUI](https://github.com/comfyanonymous/ComfyUI), but without
the UI.  It's stripped down and packaged as a library, for use in other projects.

ComfyUI is actively maintained (as of writing), and has implementations of a
lot of the cool cutting-edge Stable Diffusion stuff.  The implementation is as
unmodified as possible, to hopefully make it easier to merge in upstream
improvements.

In order to provide a consistent API, an interface layer has
been added.  Directly importing names not in the API should be considered
dangerous, as the ComfyUI maintainers might change how things are arranged
whenever they want, and those changes will be merged into this fork as soon as
possible, possibly without changing the major version of the library.

The interface layer will be consistent within a major version of the library,
so that's what you should rely on.

# Design goals

0. The implementation from ComfyUI should be changed as little as possible, and
in very predictable ways if so.  (Changing import paths, for example.)  This is
to make it easier to merge in changes from upstream, so the library can keep
apace with the work being done on the ComfyUI project.
1. The API should expose the same breadth of functionality available by using
the node editor in ComfyUI.  So, at the very least, we're probably targeting
one function/method per node.
2. Opaque types should be preferred.  Rather than pass tensors around, we're
going to wrap them in objects that hide the implementation.  This gives us
maximum flexibility to keep the API the same in case ComfyUI change things
drastically.
3. Explicit rather than implicit behavior.  ComfyUI does a lot of clever things
to provide a good user experience, like automatically rounding down sizes, or
managing VRAM.  As much as possible, we're going to try to make these explicit
options for the library-user.
4. The API should be should be typed as strictly as possible.  Enums should be
used instead of strings, when applicable, etc.
5. The interface layer should have unit tests

# Installation

For now you can install from github:

```
pip3 install git+https://github.com/adodge/Comfy-Lib
```

# Example

```python3
import comfy.stable_diffusion
import comfy.latent_image
import comfy

config = comfy.stable_diffusion.CheckpointConfig.from_built_in(comfy.stable_diffusion.BuiltInCheckpointConfigName.V1)

# Read in a checkpoint
sd, clip, vae = comfy.stable_diffusion.load_checkpoint(
    config=config,
    checkpoint_filepath="v1-5-pruned-emaonly.safetensors",
    embedding_directory=None,
)

# CLIP encode a prompt
pos = clip.encode("an astronaut")
neg = clip.encode("")

# Run the sampler
img0 = comfy.latent_image.LatentImage.empty(512, 512)
img1 = sd.sample(positive=pos, negative=neg, latent_image=img0, seed=42, steps=20, cfg_strength=7,
                 sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER, scheduler=comfy.stable_diffusion.Scheduler.NORMAL,
                 denoise_strength=1.0)

# Run the VAE to get a Pillow Image
img2 = vae.decode(latent_image=img1)

# Save that to a file
img2.save("out.png")
```

# API

*(API documentation in progress)*
