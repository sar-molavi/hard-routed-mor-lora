"""Evaluation dataset loader built on the prompt-format helpers."""

from __future__ import annotations

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .dataset import get_dataset as load_prompt_dataset


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize prompts and optionally prepend the tokenizer's BOS token."""
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["prompt"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        ),
        batched=True,
        desc="Tokenizing dataset",
    )

    return tokenized


def _ensure_label_column(dataset: Dataset, dataset_name: str) -> Dataset:
    """Reuse the shared ``ground_truth`` column but expose it as ``label``."""

    if "ground_truth" not in dataset.column_names:
        raise RuntimeError(
            f"Dataset '{dataset_name}' is missing the 'ground_truth' column after "
            "prompt formatting."
        )

    if "label" in dataset.column_names:
        dataset = dataset.remove_columns("label")

    return dataset.rename_column("ground_truth", "label")


def get_dataset(
    *,
    dataset_name: str,
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Load prompts via ``lora_trl.dataset`` and return tokenized eval inputs."""

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    prompt_dataset = load_prompt_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
    )

    if len(prompt_dataset) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' at {dataset_path} is empty after processing."
        )

    prompt_dataset = _ensure_label_column(prompt_dataset, dataset_name)
    dataset = _tokenize_dataset(prompt_dataset, tokenizer, max_length)

    keep = ["prompt", "input_ids", "label"]
    to_remove = [column for column in dataset.column_names if column not in keep]
    if to_remove:
        dataset = dataset.remove_columns(to_remove)

    return dataset
