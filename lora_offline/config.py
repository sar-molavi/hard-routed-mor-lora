"""Configuration stack for offline RL fine-tuning."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class LoraAdapterConfig:
    """
    LoRA/QLoRA hyperparameters.

    Args:
        r: Rank of the low-rank adapters.
        lora_alpha: Scaling factor applied to LoRA updates.
        lora_dropout: Dropout applied on LoRA layers.
        target_modules: Module names to wrap (e.g., attention projections).
        bias: Bias handling strategy (typically "none").
        task_type: Task identifier passed to PEFT.
        modules_to_save: Extra modules to keep trainable in addition to LoRA.
    """

    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] | None = None


@dataclass
class OfflinePrefetchConfig:
    """
    Arguments controlling how prompts/completions are fetched from the server.

    This mirrors :class:`prefetch_rlvf_dataset.PrefetchConfig` so we can build
    those objects directly via :func:`dataclasses.asdict`.

    Args:
        dataset_name: Built-in dataset key for ``dataset.get_dataset``.
        dataset_path: JSONL path when not using a built-in dataset.
        checkpoints_dir: Directory where LoRA checkpoints are written and polled.
        server_url: Base URL of the vLLM/OpenAI-compatible server.
        model_name: Model identifier exposed by the server for generation.
        api_key: API key if the server enforces auth.
        max_samples: Optional cap on total samples to consume.
        samples_per_checkpoint: Derived; samples fetched before refreshing adapters.
        request_batch_size: Derived; request batch size issued to the server.
        prefetch_queue_size: Derived; buffered queue length.
        request_timeout: Timeout (seconds) for generation calls.
        max_new_tokens: Max tokens to generate per completion.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k cutoff.
        presence_penalty: Presence penalty passed to the server.
        frequency_penalty: Frequency penalty passed to the server.
        repetition_penalty: Optional repetition penalty.
        stop_sequences: Stop strings to cut generation.
        logprobs: Top-K log probabilities to fetch per generated token.
        num_generations: Number of completions per prompt.
        generation_kwargs: Raw overrides merged into the generation payload.
        seed: Seed used for shuffling and repeats.
        n_repeat: How many shuffled passes over the dataset.
        lora_steps_per_refresh: Training steps between adapter refresh checks.
        checkpoint_min_step_delta: Minimum global step gap before switching adapter.
    """

    dataset_name: str = "math"
    dataset_path: str = ""
    checkpoints_dir: str | None = None
    server_url: str = "http://localhost:8000"
    model_name: str | None = None
    api_key: str | None = None
    max_samples: int | None = None
    # Derived values; set in OfflineTrainingConfig._apply_derived_settings
    samples_per_checkpoint: int = field(default=64, init=False)
    request_batch_size: int = field(default=4, init=False)
    prefetch_queue_size: int = field(default=64, init=False)
    request_timeout: float = 120.0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float | None = None
    stop_sequences: list[str] | None = None
    logprobs: int = 10
    num_generations: int = 1
    generation_kwargs: dict[str, Any] | None = None
    seed: int = 1997
    n_repeat: int = 3
    wait_for_new_checkpoint: bool = True
    checkpoint_wait_timeout: float | None = 180
    lora_steps_per_refresh: int = 1
    checkpoint_min_step_delta: int = 0
    allow_missing_initial_checkpoint: bool = True

    def to_prefetch_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("lora_steps_per_refresh", None)
        return data


@dataclass
class OfflineObjectiveConfig:
    """
    Loss-specific knobs shared across derived trainers.

    Args:
        loss_type: One of "sequence", "token", or "token_dapo"; selects trainer.
        ratio_clip: PPO-style lower clip.
        ratio_clip_high: Optional asymmetric upper clip (defaults to ``ratio_clip``).
        normalize_advantages: Whether to normalize advantages per batch.
        reward_scaling: Optional reward scaling mode (e.g., "group").
        advantage_mode: Advantage computation strategy (grpo/ignore/custom).
        correct_reward: Reward applied to correct answers.
        incorrect_reward: Reward applied to incorrect answers.
        bonus_think: Bonus for responses containing <think> tags.
        bonus_json_on_wrong: Bonus for incorrect answers formatted as JSON.
    """

    loss_type: Literal["sequence", "token", "token_dapo"] = "token_dapo"
    ratio_clip: float | None = 0.2
    ratio_clip_high: float | None = None
    normalize_advantages: bool = True
    reward_scaling: Literal["group"] | None = None
    advantage_mode: Literal["grpo", "ignore", "custom"] = "grpo"
    correct_reward: float = 1.0
    incorrect_reward: float = -1.1
    bonus_think: float = 0.2
    bonus_json_on_wrong: float = 0.1


@dataclass
class OfflineTrainingConfig:
    """
    Full configuration for offline RL training of LoRA adapters.

    Args:
        model_name_or_path: Base model to load (HF id or local path).
        tokenizer_name_or_path: Optional tokenizer override.
        use_lora: Enable LoRA fine-tuning.
        use_qlora: Enable QLoRA (4-bit) flow with bitsandbytes.
        lora_config: Nested ``LoraAdapterConfig``.
        output_dir: Directory for checkpoints, logs, and artifacts.
        max_sequence_length: Max sequence length used by the collator.
        per_device_train_batch_size: Per-device batch size (before dispatch scaling).
        gradient_accumulation_steps: Number of accumulation steps.
        num_train_epochs: Number of epochs when dataset size is known.
        max_steps: Optional explicit step count (derived if ``None``).
        learning_rate: Optimizer learning rate.
        lr_scheduler_type: Scheduler type for ``TrainingArguments``.
        warmup_ratio: Fraction of steps for LR warmup.
        max_grad_norm: Gradient clipping value.
        weight_decay: Weight decay factor.
        logging_steps: Logging frequency.
        save_strategy: Checkpoint strategy ("steps"/"epoch"/"no").
        save_steps: Step interval for saving when using step strategy.
        save_total_limit: Max checkpoints to retain.
        eval_strategy: Evaluation schedule ("no"/"steps"/"epoch").
        eval_steps: Eval interval when using step strategy.
        report_to: Reporting destinations (e.g., "tensorboard", "wandb").
        bf16: Enable bfloat16 mixed precision.
        fp16: Enable float16 mixed precision.
        gradient_checkpointing: Enable gradient checkpointing.
        dataloader_num_workers: DataLoader worker count.
        seed: Global seed applied in training entrypoints.
        prefetch: Nested ``OfflinePrefetchConfig``.
        algorithm: Nested ``OfflineObjectiveConfig``.
    """

    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    use_lora: bool = True
    use_qlora: bool = False
    lora_config: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)

    output_dir: str = "./lora-offline"
    max_sequence_length: int | None = None

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int | None = None
    save_total_limit: int | None = None
    eval_strategy: str = "no"
    eval_steps: int | None = None
    report_to: str | list[str] | None = "tensorboard"
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    seed: int = 42

    prefetch: OfflinePrefetchConfig = field(default_factory=OfflinePrefetchConfig)
    algorithm: OfflineObjectiveConfig = field(default_factory=OfflineObjectiveConfig)

    def __post_init__(self) -> None:
        if not self.prefetch.checkpoints_dir:
            self.prefetch.checkpoints_dir = self.output_dir
        if not self.prefetch.model_name:
            self.prefetch.model_name = self.model_name_or_path
        self._apply_derived_settings()

    @classmethod
    def from_json(cls, path: str | Path) -> "OfflineTrainingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        data["lora_config"] = LoraAdapterConfig(**data.get("lora_config", {}))
        data["prefetch"] = OfflinePrefetchConfig(**data.get("prefetch", {}))
        data["algorithm"] = OfflineObjectiveConfig(**data.get("algorithm", {}))
        return cls(**data)

    # -------------------------------------------------
    # Derived helpers
    # -------------------------------------------------
    def _apply_derived_settings(self) -> None:
        eff_batch = self._effective_batch_size()
        total_samples = self._total_samples()
        num_generations = max(int(getattr(self.prefetch, "num_generations", 1)), 1)
        if total_samples is None:
            raise ValueError(
                "Unable to determine dataset size. Provide `prefetch.max_samples` or a valid dataset_path."
            )
        epochs = max(self.num_train_epochs, 0.0)
        calculated_steps = (
            math.ceil((total_samples / eff_batch) * num_generations * epochs)
            if eff_batch > 0
            else 0
        )
        if self.max_steps is None:
            self.max_steps = calculated_steps

        steps_per_lora = max(int(self.prefetch.lora_steps_per_refresh), 1)
        prompts_per_step = max(eff_batch // num_generations, 1)
        samples_per_checkpoint = steps_per_lora * prompts_per_step
        self.prefetch.samples_per_checkpoint = samples_per_checkpoint
        self.prefetch.request_batch_size = prompts_per_step
        self.prefetch.prefetch_queue_size = max(samples_per_checkpoint, 1)

    def _effective_batch_size(self) -> int:
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
        world_size = max(world_size, 1)
        num_generations = max(int(getattr(self.prefetch, "num_generations", 1)), 1)
        return (
            max(self.per_device_train_batch_size, 1)
            * max(self.gradient_accumulation_steps, 1)
            * world_size
            * num_generations
        )

    def _total_samples(self) -> int | None:
        if self.prefetch.max_samples is not None:
            return int(self.prefetch.max_samples)
        path = Path(self.prefetch.dataset_path)
        if not path.exists():
            return None
        if path.is_file():
            return _count_jsonl_samples(path)
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
            if not files:
                return None
            return sum(_count_jsonl_samples(f) for f in files)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "use_lora": self.use_lora,
            "use_qlora": self.use_qlora,
            "lora_config": asdict(self.lora_config),
            "output_dir": self.output_dir,
            "max_sequence_length": self.max_sequence_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "report_to": self.report_to,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": self.dataloader_num_workers,
            "seed": self.seed,
            "prefetch": asdict(self.prefetch),
            "algorithm": asdict(self.algorithm),
        }


def _count_jsonl_samples(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


__all__ = [
    "OfflineTrainingConfig",
    "OfflinePrefetchConfig",
    "OfflineObjectiveConfig",
    "LoraAdapterConfig",
]
