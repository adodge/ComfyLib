from typing import Optional

from comfy.conditioning import Conditioning
from comfy.hazard.sd import CLIP, load_clip

class CLIPModel:
    def __init__(self, model: CLIP):
        self._model = model

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
        return Conditioning(result)
