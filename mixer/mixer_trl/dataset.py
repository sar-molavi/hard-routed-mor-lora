"""Dataset assembly helpers for LoRA-Mixer GRPO training."""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from ..config import DatasetConfig
from ..dataset import select_random
from lora_offline.dataset import get_dataset as load_prompt_dataset


def _standardize_grpo_columns(dataset: Dataset) -> Dataset:
    required_columns = {"prompt", "ground_truth"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise RuntimeError(
            f"Dataset is missing required GRPO columns: {sorted(missing)}"
        )

    # Keep only columns used by GRPO/reward to avoid schema conflicts across datasets.
    drop_columns = [
        name for name in dataset.column_names if name not in required_columns
    ]
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)
    return dataset


def _load_single_dataset(
    *,
    cfg: DatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    seed: int | None,
    max_samples_per_dataset: int | None,
) -> Dataset:
    dataset = load_prompt_dataset(
        dataset_name=cfg.name,
        dataset_path=cfg.fn_path,
        tokenizer=tokenizer,
    )
    dataset = _standardize_grpo_columns(dataset)

    sample_cap = cfg.max_num_fn
    if max_samples_per_dataset is not None:
        sample_cap = (
            max_samples_per_dataset
            if sample_cap is None
            else min(sample_cap, max_samples_per_dataset)
        )

    dataset = select_random(dataset=dataset, n=sample_cap, seed=seed)
    return dataset


def build_grpo_dataset(
    *,
    cfgs: list[DatasetConfig],
    tokenizer: PreTrainedTokenizerBase,
    seed: int | None = None,
    max_samples_per_dataset: int | None = None,
) -> Dataset:
    datasets: list[Dataset] = []
    for idx, cfg in enumerate(cfgs):
        dataset_seed = None if seed is None else seed + idx
        datasets.append(
            _load_single_dataset(
                cfg=cfg,
                tokenizer=tokenizer,
                seed=dataset_seed,
                max_samples_per_dataset=max_samples_per_dataset,
            )
        )

    if not datasets:
        raise ValueError("No datasets were provided for GRPO training.")

    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


__all__ = ["build_grpo_dataset"]
