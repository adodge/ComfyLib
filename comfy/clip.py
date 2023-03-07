from typing import Optional, Union

import torch

from comfy.conditioning import Conditioning
from comfy.hazard.sd import CLIP, load_clip
from comfy.util import ModelLoadError, SDType


class CLIPModel(SDType):
    def __init__(self, model: CLIP, device: Union[str, torch.device] = "cpu"):
        self._model = model
        self.to(device)

    def to(self, device: Union[str, torch.device]) -> "CLIPModel":
        """
        Modifies the object in-place.
        """
        torch_device = torch.device(device)
        if torch_device == self.device:
            return self

        self._model.cond_stage_model.to(torch_device)
        self._model.cond_stage_model.device = torch_device
        self.device = torch_device
        return self

    @classmethod
    def from_model(
        cls,
        model_filepath: str,
        stop_at_clip_layer: int = -1,
        embedding_directory: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "CLIPModel":
        # CLIPLoader

        try:
            clip = load_clip(
                ckpt_path=model_filepath, embedding_directory=embedding_directory
            )
        except Exception as e:
            raise ModelLoadError("Failed to load CLIP model.") from e
        clip.clip_layer(stop_at_clip_layer)

        return CLIPModel(clip, device=device)

    def encode(self, text: str) -> Conditioning:
        # CLIPTextEncode
        result = self._model.encode(text)
        return Conditioning(result, device=self.device)
