"""Helpers that adapt ``lora_trl`` configs to TRL's GRPOTrainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from .config import TRLTrainingConfig


def _build_trl_config(
    *,
    cfg: TRLTrainingConfig,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> GRPOConfig:
    """
    Convert :class:`RLTrainingConfig` into TRL's :class:`GRPOConfig`.
    """
    algo = cfg.algorithm
    gen = cfg.generation
    eval_batch_size = cfg.per_device_eval_batch_size or cfg.per_device_train_batch_size

    gc_kwargs = {"use_reentrant": False} if cfg.gradient_checkpointing else None
    trl_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "max_grad_norm": cfg.max_grad_norm,
        "weight_decay": cfg.weight_decay,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": cfg.eval_strategy,
        "report_to": cfg.report_to,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": gc_kwargs,
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
        "max_prompt_length": cfg.max_prompt_length or tokenizer.model_max_length,
        "max_completion_length": gen.max_new_tokens,
        "num_generations": algo.num_generations,
        "temperature": gen.temperature,
        "epsilon": algo.grpo_epsilon,
    }

    if cfg.save_strategy:
        trl_kwargs["save_strategy"] = cfg.save_strategy
    if cfg.save_steps is not None:
        trl_kwargs["save_steps"] = cfg.save_steps
    if cfg.save_total_limit is not None:
        trl_kwargs["save_total_limit"] = cfg.save_total_limit
    if cfg.eval_steps is not None:
        trl_kwargs["eval_steps"] = cfg.eval_steps
    if algo.generation_batch_size is not None:
        trl_kwargs["generation_batch_size"] = algo.generation_batch_size
    if algo.steps_per_generation is not None:
        trl_kwargs["steps_per_generation"] = algo.steps_per_generation
    if gen.top_p is not None:
        trl_kwargs["top_p"] = gen.top_p
    if gen.top_k is not None:
        trl_kwargs["top_k"] = gen.top_k
    if gen.repetition_penalty is not None:
        trl_kwargs["repetition_penalty"] = gen.repetition_penalty

    generation_kwargs: dict[str, Any] = {
        "do_sample": gen.do_sample and not gen.deterministic,
    }
    if gen.top_p is not None:
        generation_kwargs["top_p"] = gen.top_p
    if gen.top_k is not None:
        generation_kwargs["top_k"] = gen.top_k
    if gen.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = gen.repetition_penalty
    if gen.temperature is not None:
        generation_kwargs["temperature"] = gen.temperature
    trl_kwargs["generation_kwargs"] = generation_kwargs

    if algo.trl_extra_kwargs:
        trl_kwargs.update(algo.trl_extra_kwargs)

    return GRPOConfig(**trl_kwargs)


def build_grpo_trainer(
    *,
    cfg: TRLTrainingConfig,
    output_dir: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset | IterableDataset,
    reward_fn: Callable[..., Any],
) -> GRPOTrainer:
    """
    Instantiate TRL's :class:`GRPOTrainer` with project-specific wiring.
    """
    trl_config = _build_trl_config(cfg=cfg, tokenizer=tokenizer, output_dir=output_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define a pad token for TRL training.")
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=trl_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


__all__ = ["build_grpo_trainer"]
