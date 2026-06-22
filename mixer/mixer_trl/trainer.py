"""TRL GRPO trainer helpers for LoRA-Mixer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, IterableDataset
from trl import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from transformers import PreTrainedTokenizerBase

from ..utils import (
    LORA_MIXER_WEIGHTS_NAME,
    load_lora_mixer_weights,
    save_lora_mixer_weights,
    unwrap_model,
)
from ..trainer import get_router_temperature
from .config import MixerGRPOTrainingConfig


def _unwrap_mixer_model(model):
    base = unwrap_model(model)
    return getattr(base, "mixer_model", base)


class MixerGRPOTrainer(GRPOTrainer):
    """GRPOTrainer variant that persists LoRA-Mixer adapter/router checkpoints."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        router_temperature = get_router_temperature(self.model)
        if router_temperature is not None:
            logs["router_temperature"] = router_temperature
        return super().log(logs, start_time=start_time)

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        if not self.args.should_save:
            return

        target_dir = Path(output_dir or self.args.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        base_model = _unwrap_mixer_model(self.model)
        save_lora_mixer_weights(base_model, target_dir)

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str | bool | None,
        model=None,
    ):
        if isinstance(resume_from_checkpoint, bool):
            candidate = (
                self.state.best_model_checkpoint
                if resume_from_checkpoint and self.state.best_model_checkpoint
                else self.state.last_model_checkpoint
            )
            if candidate is None:
                return
            resume_from_checkpoint = candidate

        checkpoint_dir = Path(resume_from_checkpoint)
        adapter_file = checkpoint_dir / LORA_MIXER_WEIGHTS_NAME

        if adapter_file.is_file():
            base_model = _unwrap_mixer_model(model or self.model)
            load_lora_mixer_weights(base_model, checkpoint_dir, strict=False)
            return

        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)


def _build_trl_config(
    *,
    cfg: MixerGRPOTrainingConfig,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> GRPOConfig:
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
        "max_steps": cfg.max_steps,
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
        # Sparse expert routing can leave some expert params unused on a rank.
        "ddp_find_unused_parameters": not cfg.freeze_experts,
        "save_on_each_node": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": cfg.dataloader_num_workers,
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
    if cfg.deepspeed_config is not None:
        trl_kwargs["deepspeed"] = cfg.deepspeed_config
    if cfg.seed is not None:
        trl_kwargs["seed"] = cfg.seed
        trl_kwargs["data_seed"] = cfg.seed

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
    cfg: MixerGRPOTrainingConfig,
    output_dir: Path,
    model,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset | IterableDataset,
    reward_fn: Callable[..., Any],
    eval_dataset: Dataset | IterableDataset | None = None,
) -> MixerGRPOTrainer:
    trl_config = _build_trl_config(cfg=cfg, tokenizer=tokenizer, output_dir=output_dir)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define a pad token for GRPO training.")

    return MixerGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=trl_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


__all__ = ["build_grpo_trainer", "MixerGRPOTrainer"]
