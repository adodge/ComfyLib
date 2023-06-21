from enum import Enum
from typing import Optional, Union

import torch
from .conditioning import Conditioning
from .hazard.sd import CLIP, load_clip
from .util import ModelLoadError, SDType
from .hazard.sd1_clip import SD1ClipModel
from .hazard.sd2_clip import SD2ClipModel


class CLIPModelVersion(Enum):
    SD1x = "SD1.x"
    SD2x = "SD2.x"


class CLIPModel(SDType):
    def __init__(self, model: CLIP, device: Union[str, torch.device] = "cpu"):
        self._model = model
        self.to(device)

        self.version: Optional[CLIPModelVersion] = None
        if isinstance(self._model.cond_stage_model, SD2ClipModel):
            self.version = CLIPModelVersion.SD2x
        elif isinstance(self._model.cond_stage_model, SD1ClipModel):
            self.version = CLIPModelVersion.SD1x

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
