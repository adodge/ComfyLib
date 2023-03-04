from typing import Optional, Union

import torch

from comfy.conditioning import Conditioning
from comfy.hazard.sd import CLIP, load_clip


class CLIPModel:
    def __init__(self, model: CLIP, device: Union[str, torch.device] = "cpu"):
        self._model = model
        self.device: Optional[torch.device] = None
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
    ) -> "CLIPModel":
        # CLIPLoader

        clip = load_clip(
            ckpt_path=model_filepath, embedding_directory=embedding_directory
        )
        clip.clip_layer(stop_at_clip_layer)

        return CLIPModel(clip)

    def encode(self, text: str) -> Conditioning:
        # CLIPTextEncode
        result = self._model.encode(text)
        return Conditioning(result, device=self.device)
