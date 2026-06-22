"""TRL-specific configuration dataclasses and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class LoraAdapterConfig:
    """LoRA/QLoRA adapter hyperparameters."""

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
class GenerationConfig:
    """Decoding parameters forwarded to TRL's grouped rollouts."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    do_sample: bool = True
    deterministic: bool = False
    use_generate_batch: bool = False


@dataclass
class AlgorithmConfig:
    """GRPO-specific knobs consumed by TRL."""

    num_generations: int = 4
    generation_batch_size: int | None = None
    steps_per_generation: int | None = None
    grpo_epsilon: float = 1e-6
    trl_extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    """Reward parameters forwarded to `lora_offline.reward.compute_reward`."""

    uncertainty_weight: float = 0.1
    entropy_patch_length: int = 128
    entropy_patch_overlap: int = 32
    normalize_entropy: bool = True
    correct_reward: float = 1.0
    incorrect_reward: float = -1.1
    bonus_think: float = 0.2
    bonus_json_on_wrong: float = 0.1


@dataclass
class TRLTrainingConfig:
    """
    Full configuration describing TRL GRPO training for LoRA adapters.
    """

    # Model/tokenizer
    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    use_lora: bool = True
    use_qlora: bool = False
    lora_config: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)

    # Dataset + output paths
    dataset_name: str = "math"
    dataset_path: str = ""
    output_dir: str = "./trl-grpo"
    max_prompt_length: int | None = None
    pad_to_multiple_of: int | None = None

    # Trainer hyperparameters
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
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

    # Algorithm + generation + rewards
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "TRLTrainingConfig":
        """
        Load a TRLTrainingConfig from disk and validate compatibility.
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        data["lora_config"] = LoraAdapterConfig(**data.get("lora_config", {}))
        data["generation"] = GenerationConfig(**data.get("generation", {}))
        data["algorithm"] = AlgorithmConfig(**data.get("algorithm", {}))
        data["reward"] = RewardConfig(**data.get("reward", {}))

        cfg = cls(**data)
        validate_trl_config(cfg)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config into a JSON-friendly dict."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "use_lora": self.use_lora,
            "use_qlora": self.use_qlora,
            "lora_config": asdict(self.lora_config),
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "max_prompt_length": self.max_prompt_length,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
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
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "algorithm": asdict(self.algorithm),
            "generation": asdict(self.generation),
            "reward": asdict(self.reward),
        }


class TRLConfigError(ValueError):
    """Raised when TRL-specific validation fails."""


def validate_trl_config(cfg: TRLTrainingConfig) -> None:
    """Ensure config is compatible with TRL GRPO before training."""
    algo = cfg.algorithm
    if algo.num_generations < 2:
        raise TRLConfigError("GRPO requires `num_generations` >= 2.")

    if (
        algo.generation_batch_size is not None
        and algo.steps_per_generation is not None
    ):
        raise TRLConfigError(
            "`generation_batch_size` and `steps_per_generation` can not both be set."
        )

    if algo.generation_batch_size is not None:
        if algo.generation_batch_size < algo.num_generations:
            raise TRLConfigError(
                "`generation_batch_size` must be >= `num_generations`."
            )
        if algo.generation_batch_size % algo.num_generations != 0:
            raise TRLConfigError(
                "`generation_batch_size` must be divisible by `num_generations`."
            )

    if cfg.per_device_train_batch_size < 1:
        raise TRLConfigError("`per_device_train_batch_size` must be >= 1.")

    if cfg.dataset_path == "":
        raise TRLConfigError("`dataset_path` must point to a JSONL file or directory.")


__all__ = [
    "TRLTrainingConfig",
    "LoraAdapterConfig",
    "GenerationConfig",
    "AlgorithmConfig",
    "RewardConfig",
    "TRLConfigError",
    "validate_trl_config",
]
