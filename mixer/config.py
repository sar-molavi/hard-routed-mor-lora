"""Configuration dataclasses for the LoRA-Mixer training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    """
    Dataset definition for training/evaluation.

    Attributes:
        name: Dataset identifier used by dataset loaders.
        rl_path: Filesystem path to the RL-style JSONL dataset.
        fn_path: Filesystem path to the fine-tuning dataset.
        max_completion_len: Optional max completion length filter.
        max_num_traces: Optional cap on number of traces per sample.
        trace_cap_strategy: Strategy for selecting traces when capping:
            "first": Keep the first N traces (current default).
            "source_order": Sort by source priority then take first N.
            "random": Shuffle traces then take first N.
    """

    name: str
    fn_path: str = None
    rl_path: str = None
    max_completion_len: int | None = None
    max_num_traces: int | None = None
    max_num_fn: int | None = None
    max_num_rl: int | None = None
    trace_cap_strategy: str = "random"


@dataclass
class LoraAdapterConfig:
    """
    LoRA adapter hyperparameters (PEFT LoraConfig fields).

    Attributes:
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
class GumbelTemperatureSchedulerConfig:
    """
    Temperature schedule for Gumbel-Softmax routing.

    Attributes:
        name: Scheduler name. Supported values: "cosine", "exponential".
        initial_temperature: Temperature used at training step 0.
        final_temperature: Minimum temperature reached before the hold phase.
        hold_steps: Final hold duration. Values in [0, 1) are interpreted as
            a fraction of total training steps; values >= 1 are interpreted as
            an absolute number of optimizer steps.
    """

    name: str = "cosine"
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    hold_steps: float = 0.0

    def __post_init__(self) -> None:
        self.name = self.name.lower()
        if self.name not in {"cosine", "exponential"}:
            raise ValueError(
                "gumbel_temperature_scheduler.name must be one of: cosine, exponential."
            )
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")
        if self.final_temperature <= 0:
            raise ValueError("final_temperature must be positive.")
        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature must be <= initial_temperature.")
        self.hold_steps = float(self.hold_steps)
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative.")


@dataclass
class MixerTrainingConfig:
    """
    High-level configuration for LoRA-Mixer training.

    Model:
        model_name_or_path: HF model name or local path.
        expert_paths: LoRA expert checkpoints (safetensors).
        num_layers: Number of layers to patch; None uses all layers.
        enable_lora_attn: Enable attention LoRA adapters.
        normalize_router_weights: Normalize top-k router weights.
        top_k: Number of experts to route per token.
        router_alpha: Balance loss coefficient.
        jitter_noise: Optional multiplicative jitter for router inputs.
        freeze_router: Freeze router parameters.
        freeze_experts: Freeze LoRA expert parameters.

    Data:
        train_set_configs: Training dataset configs.
        eval_set_configs: Optional evaluation dataset configs.
        apply_class_weight: Apply inverse-frequency class weights.
        max_length: Optional max sequence length.

    Training:
        output_dir: Output directory for checkpoints.
        seed: Optional dataset seed for deterministic sampling/shuffling.
        per_device_train_batch_size: Train batch size per device.
        per_device_eval_batch_size: Eval batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        num_train_epochs: Number of epochs.
        max_steps: Max update steps (-1 for no cap).
        warmup_ratio: Warmup ratio.
        logging_steps: Logging frequency in steps.
        eval_strategy: HF Trainer evaluation strategy.
        eval_steps: Evaluation steps interval.
        save_strategy: HF Trainer save strategy.
        save_steps: Save interval in steps.
        save_total_limit: Max checkpoints to keep.
        report_to: Reporting backend (e.g., "tensorboard").
        fp16: Use FP16 training.
        bf16: Use BF16 training.
        gradient_checkpointing: Enable gradient checkpointing.
        remove_unused_columns: Remove unused columns from datasets.
        max_grad_norm: Gradient clipping norm.
        lr_scheduler_type: LR scheduler type.
        optim: Optimizer name.
        dataloader_num_workers: DataLoader worker count.
        load_best_model_at_end: Whether to load best checkpoint at end.
        balance_loss_weight: Weight for router auxiliary loss.
        unconstrained_experts: Expert indices exempt from distillation.
        distill_l2_reg: L2 distillation weight for LoRA experts.
        deepspeed_config: Optional DeepSpeed config JSON path (e.g. ZeRO-1).
    """

    model_name_or_path: str
    expert_paths: list[str]
    enable_lora_attn: bool
    normalize_router_weights: bool
    top_k: int
    output_dir: str
    seed: int | None = None

    num_layers: int | None = None
    router_alpha: float = 0.0
    router_gamma: float = 0.0
    router_token_gamma: float = 0.0
    router_sequence_gamma: float = 0.0
    use_default_loss: bool = True
    jitter_noise: float | None = None
    freeze_router: bool = False
    freeze_experts: bool = True
    lora_config: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)

    train_set_configs: list[DatasetConfig] = field(default_factory=list)
    eval_set_configs: list[DatasetConfig] = field(default_factory=list)
    apply_class_weight: bool = False

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.05
    logging_steps: int = 50
    eval_steps: int | None = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int | None = 3
    report_to: str | None = "tensorboard"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    remove_unused_columns: bool = False
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 4
    n_repeats: int | None = None

    balance_loss_weight: float = 0.0
    distill_l2_reg: float = 0.0
    unconstrained_experts: list[int] = field(default_factory=list)

    max_length: int | None = None
    load_best_model_at_end: bool = False
    deepspeed_config: str | None = None
    apply_hard: bool | None = None
    router_mode: str | None = None
    gumbel_hard: bool = True
    gumbel_temperature_scheduler: GumbelTemperatureSchedulerConfig = field(
        default_factory=GumbelTemperatureSchedulerConfig
    )
    router_shared_across_layers: bool | None = False

    @classmethod
    def from_json(cls, json_path: str | Path) -> "MixerTrainingConfig":
        """
        Load configuration from a JSON file.

        The JSON schema should provide a `train_set_configs` array with entries
        matching DatasetConfig fields. `eval_set_configs` is optional.
        """
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)

        payload["lora_config"] = LoraAdapterConfig(**payload.get("lora_config", {}))
        payload["gumbel_temperature_scheduler"] = GumbelTemperatureSchedulerConfig(
            **payload.get("gumbel_temperature_scheduler", {})
        )

        dataset_entries = payload.get("train_set_configs", payload.get("datasets", []))
        if not dataset_entries:
            raise ValueError("Configuration must include at least one dataset entry.")
        payload["train_set_configs"] = [
            DatasetConfig(**entry) for entry in dataset_entries
        ]

        eval_entries = payload.get("eval_set_configs", payload.get("eval_datasets", []))
        payload["eval_set_configs"] = [DatasetConfig(**entry) for entry in eval_entries]

        expert_paths = payload.get("expert_paths", [])
        if isinstance(expert_paths, str):
            expert_paths = [expert_paths]
        payload["expert_paths"] = list(expert_paths)

        unconstrained = payload.get("unconstrained_experts", [])
        if isinstance(unconstrained, int):
            unconstrained = [unconstrained]
        payload["unconstrained_experts"] = list(unconstrained)

        payload["freeze_experts"] = (
            False
            if payload.get("distill_l2_reg",0) > 0
            else payload.get("freeze_experts", True)
        )

        return cls(**payload)
