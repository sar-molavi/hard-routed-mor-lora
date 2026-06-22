"""Model/tokenizer factory helpers shared across TRL GRPO entrypoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

from .config import TRLTrainingConfig


def load_checkpoint_path(path: Path) -> str | None:
    """
    Return the latest HF checkpoint directory inside ``path`` if it exists.
    """
    if not path.is_dir():
        return None
    checkpoint_dirs = [
        candidate
        for candidate in path.iterdir()
        if candidate.is_dir() and candidate.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return None
    latest = max(checkpoint_dirs, key=lambda candidate: candidate.stat().st_mtime)
    return str(latest)


def load_tokenizer(cfg: TRLTrainingConfig) -> PreTrainedTokenizer:
    """
    Build a tokenizer with left padding and an explicit pad token.
    """
    name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must expose either a pad token or an EOS token.")
    tokenizer.padding_side = "left"
    return tokenizer


def _enable_gradient_checkpointing(model) -> None:
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()


def load_model(cfg: TRLTrainingConfig):
    """
    Instantiate LoRA/QLoRA-wrapped causal LM ready for TRL training.
    """
    model_kwargs: dict[str, Any] = {"use_cache": False}
    if cfg.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        dtype = (
            torch.bfloat16
            if cfg.bf16
            else (torch.float16 if cfg.fp16 else torch.float32)
        )
        model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    # Align the model's generation defaults with our config
    gen_cfg = model.generation_config
    gen = cfg.generation
    if gen.temperature is not None:
        gen_cfg.temperature = gen.temperature
    if gen.top_p is not None:
        gen_cfg.top_p = gen.top_p
    if gen.top_k is not None:
        gen_cfg.top_k = gen.top_k
    if gen.repetition_penalty is not None:
        gen_cfg.repetition_penalty = gen.repetition_penalty
    gen_cfg.do_sample = gen.do_sample

    if cfg.use_lora:
        if cfg.use_qlora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
            )
            if cfg.gradient_checkpointing:
                _enable_gradient_checkpointing(model)
        elif cfg.gradient_checkpointing:
            _enable_gradient_checkpointing(model)

        peft_config = LoraConfig(**asdict(cfg.lora_config))
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if cfg.gradient_checkpointing and hasattr(model, "_set_static_graph"):
        try:
            model._set_static_graph()
        except Exception:
            pass

    return model


__all__ = ["load_checkpoint_path", "load_model", "load_tokenizer"]
