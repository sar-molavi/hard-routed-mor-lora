"""Tokenized dataset loader for prepared mixed-evaluation JSONL files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> Dataset:
    logger.info("Reading prepared mixed dataset: %s", path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "prompt" not in payload or "label" not in payload:
                raise ValueError(
                    f"Mixed eval record at {path}:{line_number} must contain "
                    "'prompt' and 'label'."
                )
            records.append(payload)
    if not records:
        raise ValueError(f"Mixed eval dataset is empty: {path}")
    logger.info("Loaded %d prepared mixed records", len(records))
    return Dataset.from_list(records)


def _apply_chat_or_completion_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    if getattr(tokenizer, "chat_template", None) is not None:
        logger.info("Applying tokenizer chat template")

        def apply_chat(batch: dict[str, list[str]]) -> dict[str, list[str]]:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch["prompt"]
            ]
            return {"prompt": prompts}

        return dataset.map(apply_chat, batched=True, desc="Applying chat template")

    bos_token = tokenizer.bos_token or ""
    logger.info("Applying completion template; bos_token_present=%s", bool(bos_token))

    def apply_completion(example: dict[str, Any]) -> dict[str, str]:
        return {"prompt": bos_token + example["prompt"] + "\n\n---\n\nAnswer:\n"}

    return dataset.map(apply_completion, desc="Applying completion template")


def get_dataset(
    *,
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Load a prepared mixed JSONL file and tokenize prompts."""
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    dataset = _read_jsonl(Path(dataset_path))
    dataset = _apply_chat_or_completion_template(dataset, tokenizer)
    logger.info("Tokenizing %d mixed prompts", len(dataset))
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["prompt"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        ),
        batched=True,
        desc="Tokenizing mixed dataset",
    )

    too_long = [
        idx for idx, ids in enumerate(tokenized["input_ids"]) if len(ids) > max_length
    ]
    if too_long:
        first = too_long[0]
        raise ValueError(
            f"{len(too_long)} mixed prompts exceed max_length={max_length}; "
            f"first long record index={first}, length={len(tokenized[first]['input_ids'])}."
        )
    logger.info("Tokenized mixed dataset; max_length=%d", max_length)

    keep = ["prompt", "input_ids", "label"]
    to_remove = [column for column in tokenized.column_names if column not in keep]
    if to_remove:
        tokenized = tokenized.remove_columns(to_remove)
    return tokenized
