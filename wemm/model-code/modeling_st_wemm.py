"""Sentence Transformers module for WeMM-Embedding.

Reproduces the `transformers` usage from the model card inside a Sentence Transformers
pipeline: vision inputs are prepared with `qwen_vl_utils.process_vision_info` and the
embedding is read from `WeMMEmbedding.embedding`, which pools the `<embedding>` position and
L2-normalizes. Everything else (batching, prompts, truncation, `encode_query` /
`encode_document`, similarity) comes from the stock `Transformer` module.
"""

from __future__ import annotations

import inspect
from typing import Any

from sentence_transformers.models import Transformer


class WeMMTransformer(Transformer):
    """`Transformer` that prepares images and videos the way the model card's snippet does."""

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        super().__init__(model_name_or_path, **kwargs)
        vision_config = getattr(self.config, "vision_config", None)
        self.image_patch_size = int(getattr(vision_config, "patch_size", 16))

        # `embedding` hands its **kwargs to the inner model, so filtering on its own signature
        # would drop `pixel_values`. Filter on the inner model's parameters instead, plus the
        # processor's input names for anything the model only accepts as **kwargs.
        inner_model = getattr(self.model, "model", self.model)
        signature = set(inspect.signature(inner_model.forward).parameters)
        signature |= set(getattr(self.processor, "model_input_names", ()))
        for modality_params in self.modality_config.values():
            method_name = modality_params["method"]
            if method_name != "forward":
                self._method_signature_cache.setdefault(method_name, signature)

    def _apply_chat_template(
        self,
        messages: list[list[dict[str, Any]]],
        modality_kwargs: dict[str, dict[str, Any]],
        common_kwargs: dict[str, Any],
        chat_template_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Render the chat template and prepare images / videos exactly as the model card does."""
        from qwen_vl_utils import process_vision_info

        chat_template_kwargs = {"add_generation_prompt": False, **chat_template_kwargs}
        texts = [
            self.processor.apply_chat_template(conversation, tokenize=False, **chat_template_kwargs)
            for conversation in messages
        ]
        images, videos, video_kwargs = process_vision_info(
            [list(conversation) for conversation in messages],
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = (list(part) for part in zip(*videos))
            video_kwargs = {**video_kwargs, "video_metadata": video_metadata}

        return self.processor(
            text=texts,
            images=images,
            videos=videos,
            text_kwargs=modality_kwargs["text"],
            images_kwargs=modality_kwargs["image"],
            videos_kwargs={**modality_kwargs["video"], **video_kwargs},
            common_kwargs=common_kwargs,
        )
