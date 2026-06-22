from __future__ import annotations


from dataclasses import dataclass
import json


@dataclass
class DatasetInfo:
    name: str
    path: str


@dataclass
class FineTuningConfig:
    model_id: str
    output_dir: str
    dataset_info: list[DatasetInfo]
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    save_steps: int
    eval_steps: int
    save_total_limit: int
    logging_steps: int
    max_length: int
    max_grad_norm: float
    bf16: bool
    early_stopping_patience: int
    max_train_samples: int = None
    validation_dataset_info: list[DatasetInfo] = None
    save_strategy: str = "steps"
    load_best_model_at_end: bool = False
    eval_strategy: str = "no"
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    report_to: str = "tensorboard"

    @classmethod
    def from_dict(cls, config_dict: dict) -> FineTuningConfig:
        config_dict["dataset_info"] = [
            DatasetInfo(**info) for info in config_dict["dataset_info"]
        ]
        if config_dict.get("validation_dataset_info"):
            config_dict["validation_dataset_info"] = [
                DatasetInfo(**info)
                for info in config_dict["validation_dataset_info"]
            ]
        return cls(**config_dict)

    @classmethod
    def from_json(cls, config_path: str) -> FineTuningConfig:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
