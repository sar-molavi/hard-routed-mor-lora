"""
Evaluation dataset helpers aligned with the training prompt formatting.

`get_dataset` loads a JSONL file, formats prompts using `prompts.py` templates,
and returns a Hugging Face `Dataset` containing `prompt`, `input_ids`, and
`label` columns suitable for vLLM evaluation.
"""

from __future__ import annotations

from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .dataset import FORMATTERS, _load_json_dataset


def _apply_chat_template(
    *,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None,
    enable_thinking: bool,
) -> Dataset:
    """Format prompts using the tokenizer chat template."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise ValueError(
            "Tokenizer must implement `apply_chat_template` to format chat prompts."
        )

    def _format_prompt(example: dict[str, str]) -> dict[str, list[int]]:
        messages = [{"role": "user", "content": example["prompt"]}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if max_length is not None:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids}

    return dataset.map(_format_prompt, desc="Applying chat template")


def _apply_pretrained_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None,
) -> Dataset:
    """
    Wrap plain-text prompts with the tokenizer.
    """

    def _apply_template(example: dict[str, str]) -> dict[str, str]:
        prompt = example["prompt"]
        prompt += "\n\n---\n\nAnswer:\n"

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if max_length is not None:
            input_ids = input_ids[:max_length]

        input_ids = [tokenizer.bos_token_id] + input_ids

        return {"input_ids": input_ids}

    dataset = dataset.map(
        _apply_template,
        desc="Formatting chat prompts",
    )

    return dataset


def get_dataset(
    *,
    dataset_name: str,
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    enable_thinking: bool = False,
) -> Dataset:
    """
    Load and tokenize an evaluation dataset identified by ``dataset_name``.

    Args:
        dataset_name: One of the keys in ``FORMATTERS``.
        dataset_path: Path to the raw dataset JSONL file.
        tokenizer: Tokenizer used to convert text into model inputs.
        max_length: Maximum prompt length after tokenization (truncate if needed).
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    tokenizer.padding_side = "left"

    name = dataset_name.lower()
    if name not in FORMATTERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Available options: {', '.join(sorted(FORMATTERS))}"
        )

    dataset = _load_json_dataset(Path(dataset_path))
    dataset = FORMATTERS[name](dataset)

    required_columns = {"prompt", "label"}
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in required_columns]
    )
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise RuntimeError(
            f"Dataset '{dataset_name}' is missing columns: {sorted(missing)}"
        )

    if getattr(tokenizer, "chat_template", None):
        dataset = _apply_chat_template(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            enable_thinking=enable_thinking,
        )
    else:
        print("Ordinary template")
        dataset = _apply_pretrained_template(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    keep = ["prompt", "input_ids", "label"]
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in keep]
    )

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' at {dataset_path} is empty after processing."
        )

    return dataset


__all__ = ["get_dataset"]
