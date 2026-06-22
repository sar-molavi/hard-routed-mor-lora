from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer
from datasets import Dataset, concatenate_datasets

from .config import DatasetInfo

LABEL2KEY = {
    "medqa": "question",
    "gsm8k": "question",
    "cola": "sentence",
    "arc": "question",
    "boolq": "question",
}

LABEL2ID = {label: i for i, label in enumerate(LABEL2KEY)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def _read_json_using_json(*, path: str | Path, label: str) -> Dataset:
    records = []
    key = LABEL2KEY[label]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            records.append({"text": line[key], "label": label})
    return Dataset.from_list(records)


def get_dataset(
    info: list[DatasetInfo],
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """
    Load a raw JSONL datasets.
    """
    dataset_names = [info.name for info in info]
    dataset_paths = [info.path for info in info]

    if len(dataset_names) != len(dataset_paths):
        raise ValueError("dataset_names and dataset_paths must have the same length.")

    if not all(name in LABEL2KEY for name in dataset_names):
        raise ValueError("dataset_names must be in LABEL2KEY.")

    datasets = []
    for name, path in zip(dataset_names, dataset_paths):
        dataset = _read_json_using_json(path=path, label=name)
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError("max_samples must be non-negative.")
            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        datasets.append(dataset)
    return concatenate_datasets(datasets)


@dataclass
class DatasetCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int = None

    def __call__(self, batch):
        text = [f["text"] for f in batch]
        label_id = [LABEL2ID[f["label"]] for f in batch]

        label_id = torch.tensor(label_id, dtype=torch.long)

        inputs = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs["labels"] = label_id
        return inputs
