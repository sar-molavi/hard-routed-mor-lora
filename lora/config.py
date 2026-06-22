"""
SFT/config.py

Defines the configuration for the fine-tuning process using Pydantic dataclasses.

This module provides a structured and type-safe way to manage hyperparameters
for the training script. The configuration is loaded from a JSON file and parsed
into a nested dataclass structure.
"""

import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LoraConfigData:
    """Dataclass for LoRA (Low-Rank Adaptation) specific parameters."""

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
class TrainingConfig:
    """
    Main configuration class for the fine-tuning script.

    This dataclass holds all the hyperparameters and settings needed for the
    training process, from model and dataset paths to training arguments and
    LoRA settings.
    """

    # Model and tokenizer parameters
    model_name_or_path: str

    # LoRA and QLoRA settings
    use_lora: bool
    use_qlora: bool  # Flag to enable 4-bit quantization. Default False
    lora_config: LoraConfigData

    # Dataset and output paths
    dataset_name: str | list[str]
    dataset_path: str | list[str]
    output_dir: str
    validation_dataset_path: str | list[str] | None
    load_best_model_at_end: bool
    seed: int | None
    max_training_sample: int | None

    # Training hyperparameters
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    max_grad_norm: float
    weight_decay: float
    early_stopping_patience: int

    # Logging and saving settings
    save_strategy: str  # Default epoch
    eval_strategy: str  # Default no
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int

    # Precision and performance settings
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool

    # Reporting and logging
    report_to: str = "tensorboard"

    # DeepSpeed configuration (path to JSON config). None disables DeepSpeed.
    deepspeed_config_path: str | None = None

    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """
        Loads configuration from a JSON file and creates an instance of TrainingConfig.

        The method reads the specified JSON file, and its structure should match the
        fields defined in this dataclass, including the nested LoraConfigData.

        Args:
            json_path (str): The path to the JSON configuration file.

        Returns:
            TrainingConfig: An instance of the TrainingConfig class populated with
                            data from the JSON file.
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        # Set default values for fields that might be missing
        config_dict["per_device_eval_batch_size"] = config_dict.get(
            "per_device_eval_batch_size", 1
        )
        config_dict["use_qlora"] = config_dict.get("use_qlora", False)
        config_dict["learning_rate"] = config_dict.get("learning_rate", 1e-6)
        config_dict["lr_scheduler_type"] = config_dict.get(
            "lr_scheduler_type", "cosine"
        )
        config_dict["max_grad_norm"] = config_dict.get("max_grad_norm", 0.5)
        config_dict["weight_decay"] = config_dict.get("weight_decay", 0.01)
        config_dict["early_stopping_patience"] = config_dict.get(
            "early_stopping_patience", 10
        )
        config_dict["warmup_ratio"] = config_dict.get("warmup_ratio", 0.05)
        config_dict["save_strategy"] = config_dict.get("save_strategy", "epoch")
        config_dict["eval_strategy"] = config_dict.get("eval_strategy", "no")
        config_dict["eval_steps"] = config_dict.get("eval_steps", None)
        config_dict["save_steps"] = config_dict.get("save_steps", 100)
        config_dict["save_total_limit"] = config_dict.get("save_total_limit", None)
        config_dict["deepspeed_config_path"] = config_dict.get(
            "deepspeed_config_path", None
        )

        # Dataset and output paths
        config_dict["validation_dataset_path"] = config_dict.get(
            "validation_dataset_path", None
        )
        config_dict["load_best_model_at_end"] = config_dict.get(
            "load_best_model_at_end", False
        )
        config_dict["seed"] = config_dict.get("seed", None)
        config_dict["max_training_sample"] = config_dict.get(
            "max_training_sample", None
        )

        # Create LoraConfigData instance from the nested dictionary
        if "lora_config" in config_dict:
            lora_config_data = LoraConfigData(**config_dict.pop("lora_config"))
        else:
            lora_config_data = LoraConfigData()

        # Create TrainingConfig instance
        return cls(lora_config=lora_config_data, **config_dict)
