# This is a work in progress

All the talk about having a reliable interface below is aspirational.  The
interface is in flux at the moment, and no guarantee of the master branch
working are made, yet.  Soon.

This was initially planned to be a fork of ComfyUI, with regular merges from
upstream.  However, we are quickly realizing that's impractical, given the
rapid development in the ComfyUI codebase.  So, this is a hard fork.  We will
make a strong effort to keep parity, but we will also be going our own way here
and the implementation will probably end up quite different.

----

This is the [ComfyUI](https://github.com/comfyanonymous/ComfyUI), but without
the UI.  It's stripped down and packaged as a library, for use in other projects.

ComfyUI is actively maintained (as of writing), and has implementations of a
lot of the cool cutting-edge Stable Diffusion stuff.

In order to provide a consistent API, an interface layer has
been added.  Directly importing names not in the API should be considered
dangerous.  A best effort will be made to keep this library apace with the
implementation in ComfyUI (though this will get harder over time as the
implemntations diverge) so the backend implementation might change drastically
between minor versions.

The interface layer will be consistent within a major version of the library,
so that's what you should rely on.

# Design goals

1. The API should expose the same breadth of functionality available by using
the node editor in ComfyUI.
2. Opaque types should be preferred.  Rather than pass tensors around, we're
going to wrap them in objects that hide the implementation.  This gives us
maximum flexibility to keep the API the same, while also incorporating new
developments.
3. Explicit rather than implicit behavior.  As a library, we shouldn't make
assumptions about how the user wants to, for example, sanitize inputs or manage
VRAM.
4. The API should be should be typed as strictly as possible.  Enums should be
used instead of strings, when applicable, etc.
5. The interface layer should have complete test coverage, and the tests should
double as example code for using the library.

# Installation

For now you can install from github:

```
pip3 install git+https://github.com/adodge/ComfyLib
```

# Example

```python3
import comfy.stable_diffusion
import comfy.latent_image

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
latent0 = comfy.latent_image.LatentImage.empty(512, 512)
latent1 = sd.sample(positive=pos, negative=neg, latent_image=latent0, seed=42, steps=20, cfg_strength=7,
                 sampler=comfy.stable_diffusion.Sampler.SAMPLE_EULER, scheduler=comfy.stable_diffusion.Scheduler.NORMAL,
                 denoise_strength=1.0)

# Run the VAE to get a Pillow Image
image = vae.decode(latent_image=latent1)

# Save that to a file
image.save("out.png")
```

# API

*(API documentation in progress)*
