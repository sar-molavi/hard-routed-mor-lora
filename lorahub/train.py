from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from .algorithm import lorahub_learning


@dataclass
class DatasetConfig:
    """
    Dataset definition for training/evaluation.

    Attributes:
        name:
            Dataset identifier used by dataset loaders.
        fn_path:
            Optional filesystem path to the supervised fine-tuning dataset.
        rl_path:
            Optional filesystem path to the RL-style JSONL dataset.
        max_completion_len:
            Optional maximum completion length filter.
        max_num_traces:
            Optional cap on traces per sample.
        max_num_fn:
            Optional cap on the number of supervised fine-tuning examples.
        max_num_rl:
            Optional cap on the number of RL-style examples.
        trace_cap_strategy:
            Strategy used when `max_num_traces` is applied.
    """

    name: str
    fn_path: str | None = None
    rl_path: str | None = None
    max_completion_len: int | None = None
    max_num_traces: int | None = None
    max_num_fn: int | None = None
    max_num_rl: int | None = None
    trace_cap_strategy: str = "random"


@dataclass
class TrainConfig:
    """
    Top-level configuration for LoRAHub-style expert merging.
    """

    model_name_or_path: str
    expert_paths: list[str]
    train_set_configs: list[DatasetConfig]

    output_dir: str = "lorahub_output"
    batch_size: int = 128
    max_steps: int = 40
    l1_regularization: float = 0.05
    weight_bound: float | None = 1.5
    torch_dtype: str = "auto"
    seed: int = 42
    num_workers: int = 0

    # Checkpointing options
    checkpoint_dir: str | None = None
    resume: bool = False
    checkpoint_interval: int = 1

    save_history: bool = True


def load_config(path: str) -> TrainConfig:
    """
    Load JSON config and materialize nested `train_set_configs` entries
    into `DatasetConfig` dataclass objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    raw_dataset_cfgs = cfg_dict.get("train_set_configs", [])
    dataset_cfgs = [DatasetConfig(**item) for item in raw_dataset_cfgs]
    cfg_dict["train_set_configs"] = dataset_cfgs

    return TrainConfig(**cfg_dict)


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    """
    Save JSON with UTF-8 encoding and indentation.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to JSON config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dir(config.output_dir)

    best_weights, model, tokenizer, history = lorahub_learning(config)

    merged_lora_dir = os.path.join(config.output_dir, "merged_lora")
    ensure_dir(merged_lora_dir)

    # Save the merged adapter in PEFT format.
    # This is the artifact to load later for standard PEFT usage.
    model.save_pretrained(
        merged_lora_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(merged_lora_dir)

    save_json(
        os.path.join(config.output_dir, "final_weights.json"),
        {"weights": best_weights.tolist()},
    )

    save_json(
        os.path.join(config.output_dir, "resolved_config.json"),
        asdict(config),
    )

    if config.save_history:
        save_json(
            os.path.join(config.output_dir, "optimization_history.json"),
            history,
        )

    print("Training finished.")
    print(f"Merged LoRA adapter saved to: {merged_lora_dir}")


if __name__ == "__main__":
    main()
