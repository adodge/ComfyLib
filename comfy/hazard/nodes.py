import torch

import comfy.hazard.samplers
from comfy.hazard import model_management
from hazard.ldm.models.diffusion.ddpm import LatentDiffusion


def common_ksampler(device, model: LatentDiffusion, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str,
                    positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False):
    latent_image = latent["samples"]
    noise_mask = None

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=device)
    else:
        g_cuda = torch.Generator(device='cuda')
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=g_cuda.manual_seed(seed), device=device)

    if "noise_mask" in latent:
        noise_mask = latent['noise_mask']
        noise_mask = torch.nn.functional.interpolate(noise_mask[None,None,], size=(noise.shape[2], noise.shape[3]), mode="bilinear")
        noise_mask = noise_mask.round()
        noise_mask = torch.cat([noise_mask] * noise.shape[1], dim=1)
        noise_mask = torch.cat([noise_mask] * noise.shape[0])

    noise = noise.to(device)

    positive_copy = []
    negative_copy = []

    control_nets = []
    for p in positive:
        t = p[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        if 'control' in p[1]:
            control_nets += [p[1]['control']]
        positive_copy += [[t] + p[1:]]
    for n in negative:
        t = n[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        negative_copy += [[t] + n[1:]]

    control_net_models = []
    for x in control_nets:
        control_net_models += x.get_control_models()
    if control_net_models:
        print("Controlnet models not loaded: ", control_net_models)
    # model_management.load_controlnet_gpu(control_net_models)

    sampler = comfy.hazard.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask)
    for c in control_nets:
        c.cleanup()

    out = latent.copy()
    out["samples"] = samples
    return (out, )
