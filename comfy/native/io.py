import torch
import safetensors.torch
from typing import Dict, Tuple, Optional
from ..hazard.sd import VAE, CLIP
from ..hazard.sd import load_model_from_config
from ..hazard.ldm.models.diffusion.ddpm import LatentDiffusion


def load_torch_file(filepath: str) -> Dict:
    """
    Load a torch checkpoint file, returning the state dict.
    """
    if filepath.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(filepath, device="cpu")
    else:
        pl_sd = torch.load(filepath, map_location="cpu")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def load_checkpoint(config: Dict, filepath: str, embedding_directory: Optional[str]=None) -> Tuple[LatentDiffusion, CLIP, VAE]:
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']
    scale_factor = model_config_params['scale_factor']
    vae_config = model_config_params['first_stage_config']

    class WeightsLoader(torch.nn.Module):
        pass

    w = WeightsLoader()
    load_state_dict_to = [w]

    vae = VAE(scale_factor=scale_factor, config=vae_config)
    w.first_stage_model = vae.first_stage_model
    clip = CLIP(config=clip_config, embedding_directory=embedding_directory)
    w.cond_stage_model = clip.cond_stage_model

    model = load_model_from_config(config, filepath, verbose=False, load_state_dict_to=load_state_dict_to)
    return model, clip, vae
