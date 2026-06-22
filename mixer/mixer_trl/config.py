"""Configuration dataclasses for GRPO training with LoRA-Mixer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import (
    DatasetConfig,
    GumbelTemperatureSchedulerConfig,
    LoraAdapterConfig,
)


@dataclass
class GenerationConfig:
    """Decoding parameters forwarded to grouped rollouts."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    do_sample: bool = True
    deterministic: bool = False


@dataclass
class AlgorithmConfig:
    """GRPO-specific options."""

    num_generations: int = 4
    generation_batch_size: int | None = None
    steps_per_generation: int | None = None
    grpo_epsilon: float = 1e-6
    trl_extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    """Reward parameters forwarded to `lora_offline.reward.compute_reward`."""

    name: str = "exact_match"
    correct_reward: float = 1.0
    incorrect_reward: float = -1.1
    bonus_think: float = 0.2
    bonus_json_on_wrong: float = 0.1


@dataclass
class MixerGRPOTrainingConfig:
    """Unified config for LoRA-Mixer GRPO training."""

    model_name_or_path: str
    expert_paths: list[str]
    output_dir: str
    train_set_configs: list[DatasetConfig]

    enable_lora_attn: bool = True
    normalize_router_weights: bool = False
    top_k: int = 1
    num_layers: int | None = None
    router_alpha: float = 0.0
    router_token_gamma: float = 0.0
    router_sequence_gamma: float = 0.0
    jitter_noise: float | None = None
    freeze_router: bool = False
    freeze_experts: bool = True
    lora_config: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)
    apply_hard: bool | None = None
    router_mode: str | None = None
    gumbel_hard: bool = True
    gumbel_temperature_scheduler: GumbelTemperatureSchedulerConfig = field(
        default_factory=GumbelTemperatureSchedulerConfig
    )
    router_shared_across_layers: bool = False

    seed: int | None = None
    max_length: int | None = None
    max_samples_per_dataset: int | None = None

    max_prompt_length: int | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_strategy: str | None = "steps"
    save_steps: int | None = None
    save_total_limit: int | None = None
    eval_strategy: str = "no"
    eval_steps: int | None = None
    report_to: str | list[str] | None = "tensorboard"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    deepspeed_config: str | None = None

    eval_set_configs: list[DatasetConfig] = field(default_factory=list)
    init_mixer_checkpoint: str | None = None

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "MixerGRPOTrainingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        payload["lora_config"] = LoraAdapterConfig(**payload.get("lora_config", {}))
        payload["gumbel_temperature_scheduler"] = GumbelTemperatureSchedulerConfig(
            **payload.get("gumbel_temperature_scheduler", {})
        )
        payload["algorithm"] = AlgorithmConfig(**payload.get("algorithm", {}))
        payload["generation"] = GenerationConfig(**payload.get("generation", {}))
        payload["reward"] = RewardConfig(**payload.get("reward", {}))

        train_entries = payload.get("train_set_configs", payload.get("datasets", []))
        if not train_entries:
            raise ValueError("At least one dataset entry is required in train_set_configs.")
        payload["train_set_configs"] = [DatasetConfig(**entry) for entry in train_entries]

        eval_entries = payload.get("eval_set_configs", payload.get("eval_datasets", []))
        payload["eval_set_configs"] = [DatasetConfig(**entry) for entry in eval_entries]

        expert_paths = payload.get("expert_paths", [])
        if isinstance(expert_paths, str):
            expert_paths = [expert_paths]
        payload["expert_paths"] = list(expert_paths)

        cfg = cls(**payload)
        validate_config(cfg)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["train_set_configs"] = [asdict(entry) for entry in self.train_set_configs]
        data["eval_set_configs"] = [asdict(entry) for entry in self.eval_set_configs]
        return data


class MixerGRPOConfigError(ValueError):
    """Raised when GRPO mixer config is invalid."""


def validate_config(cfg: MixerGRPOTrainingConfig) -> None:
    if cfg.algorithm.num_generations < 2:
        raise MixerGRPOConfigError("GRPO requires num_generations >= 2.")

    if (
        cfg.algorithm.generation_batch_size is not None
        and cfg.algorithm.steps_per_generation is not None
    ):
        raise MixerGRPOConfigError(
            "generation_batch_size and steps_per_generation cannot both be set."
        )

    if cfg.algorithm.generation_batch_size is not None:
        if cfg.algorithm.generation_batch_size < cfg.algorithm.num_generations:
            raise MixerGRPOConfigError(
                "generation_batch_size must be >= num_generations."
            )
        if cfg.algorithm.generation_batch_size % cfg.algorithm.num_generations != 0:
            raise MixerGRPOConfigError(
                "generation_batch_size must be divisible by num_generations."
            )

    if cfg.algorithm.steps_per_generation is not None and cfg.algorithm.steps_per_generation <= 0:
        raise MixerGRPOConfigError("steps_per_generation must be positive.")

    if not cfg.expert_paths:
        raise MixerGRPOConfigError("expert_paths must contain at least one checkpoint path.")

    if cfg.per_device_train_batch_size < 1:
        raise MixerGRPOConfigError("per_device_train_batch_size must be >= 1.")

    if cfg.max_samples_per_dataset is not None and cfg.max_samples_per_dataset <= 0:
        raise MixerGRPOConfigError("max_samples_per_dataset must be positive.")

    for ds_cfg in cfg.train_set_configs:
        if not ds_cfg.fn_path:
            raise MixerGRPOConfigError(
                f"Dataset '{ds_cfg.name}' requires fn_path for GRPO prompt training."
            )


__all__ = [
    "AlgorithmConfig",
    "GenerationConfig",
    "MixerGRPOConfigError",
    "MixerGRPOTrainingConfig",
    "RewardConfig",
    "validate_config",
]
