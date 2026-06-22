"""HF-compatible adapter for LoRA-Mixer models used by TRL trainers."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel


class HFCompatibleMixerModel(PreTrainedModel):
    """
    Thin adapter that makes a LoRA-Mixer nn.Module look like a HF PreTrainedModel.
    """

    base_model_prefix = "mixer_model"

    def __init__(self, mixer_model: torch.nn.Module):
        config = getattr(mixer_model, "wrapped_model", mixer_model).config
        super().__init__(config)
        self.mixer_model = mixer_model
        self.generation_config = getattr(
            getattr(mixer_model, "wrapped_model", mixer_model),
            "generation_config",
            None,
        )
        self.name_or_path = getattr(
            getattr(mixer_model, "wrapped_model", mixer_model),
            "name_or_path",
            getattr(config, "_name_or_path", ""),
        )

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(
            getattr(self.mixer_model, "enable_gradient_checkpointing", False)
            or getattr(self.mixer_model, "is_gradient_checkpointing", False)
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        if hasattr(self.mixer_model, "gradient_checkpointing_enable"):
            self.mixer_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.mixer_model, "gradient_checkpointing_disable"):
            self.mixer_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        if wrapped is not None and hasattr(wrapped, "enable_input_require_grads"):
            wrapped.enable_input_require_grads()

    def get_input_embeddings(self):
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        if wrapped is not None and hasattr(wrapped, "get_input_embeddings"):
            return wrapped.get_input_embeddings()
        return None

    def set_input_embeddings(self, value) -> None:
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        if wrapped is not None and hasattr(wrapped, "set_input_embeddings"):
            wrapped.set_input_embeddings(value)

    def get_output_embeddings(self):
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        if wrapped is not None and hasattr(wrapped, "get_output_embeddings"):
            return wrapped.get_output_embeddings()
        return None

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None):
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        if wrapped is not None and hasattr(wrapped, "resize_token_embeddings"):
            return wrapped.resize_token_embeddings(
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        return None

    def can_generate(self) -> bool:
        if not hasattr(self, "mixer_model"):
            return False
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        return bool(wrapped is not None and hasattr(wrapped, "generate"))

    def generate(self, *args, **kwargs):
        wrapped = getattr(self.mixer_model, "wrapped_model", None)
        if wrapped is None or not hasattr(wrapped, "generate"):
            raise AttributeError("Wrapped model does not expose `generate`.")
        return wrapped.generate(*args, **kwargs)

    def add_model_tags(self, _tags) -> None:
        # Best-effort no-op for TRL integration.
        return None

    def get_base_model(self):
        return self.mixer_model

    def forward(self, *args, **kwargs):
        return self.mixer_model(*args, **kwargs)
