from __future__ import annotations

from pathlib import Path
import json
import random
from functools import partial
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, concatenate_datasets, DatasetDict
from transformers import PreTrainedTokenizerBase

from .config import DatasetConfig
from lora.dataset import get_dataset as get_fine_tuning_dataset

#####################

SCHEMA = """
<think>
{reasoning}
</think>

{{
  "answer": "{answer}"
}}
""".strip()


def _load_json_dataset(path: str | Path) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def _make_completion(example: dict) -> dict:
    reasoning = example["reasoning"] or example["cot"]
    answer = example["answer"] or example["response"]
    return {
        "completion": SCHEMA.format(reasoning=reasoning.strip(), answer=answer.strip())
    }


def _cap_traces_per_sample(example: dict, cap: int) -> dict:
    n_traces = len(example["cots"])
    if cap >= n_traces:
        return {}

    keys = ["cots", "reasonings", "answers", "responses"]
    return {key: example[key][:cap] for key in keys}


def _cap_traces_per_sample_by_source_order(
    example: dict,
    cap: int,
    source_order: list[str] | None = None,
) -> dict:
    n_traces = len(example["cots"])
    if cap >= n_traces:
        return {}

    if "sources" not in example:
        return _cap_traces_per_sample(example, cap)

    if source_order is None:
        source_order = ["rl_greedy", "rl", "base_line", "base_line_greedy"]

    priorities = {name: idx for idx, name in enumerate(source_order)}
    indices = list(range(n_traces))
    indices.sort(
        key=lambda i: (priorities.get(example["sources"][i], len(priorities)), i)
    )

    selected = indices[:cap]
    keys = ["cots", "reasonings", "answers", "responses", "sources"]
    return {key: [example[key][i] for i in selected] for key in keys if key in example}


def _cap_traces_per_sample_random(
    example: dict,
    cap: int,
    rng: random.Random | None = None,
) -> dict:
    n_traces = len(example["cots"])
    if cap >= n_traces:
        return {}

    if rng is None:
        rng = random

    indices = list(range(n_traces))
    rng.shuffle(indices)

    selected = indices[:cap]
    keys = ["cots", "reasonings", "answers", "responses", "sources"]
    return {key: [example[key][i] for i in selected] for key in keys if key in example}


def _explode_batch(batch: dict[str, list]) -> dict:
    out = {
        "index": [],
        "input_ids": [],
        "label": [],
        "cot": [],
        "reasoning": [],
        "answer": [],
        "response": [],
    }

    for i in range(len(batch["index"])):
        n = len(batch["cots"][i])

        for j in range(n):
            out["index"].append(batch["index"][i])
            out["input_ids"].append(batch["input_ids"][i])
            out["label"].append(batch["label"][i])
            ######
            out["cot"].append(batch["cots"][i][j])
            out["reasoning"].append(batch["reasonings"][i][j])
            out["answer"].append(batch["answers"][i][j])
            out["response"].append(batch["responses"][i][j])

    return out


def _tokenize(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_size: int | None
) -> Dataset:
    def _apply_tokenization(examples: dict[str | list]) -> dict[str | list]:
        completion_ids = tokenizer(
            examples["completion"],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        for c_ids in completion_ids:
            c_ids.append(tokenizer.eos_token_id)

        return {"completion_ids": completion_ids}

    def _make_labels(example: dict) -> dict:
        input_ids_len = len(example["input_ids"])
        input_ids = example["input_ids"] + example["completion_ids"]

        labels = input_ids.copy()
        for i in range(input_ids_len):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(_apply_tokenization, batched=True)

    if max_size:
        dataset = dataset.filter(
            lambda example: len(example["completion_ids"]) <= max_size
        )

    dataset = dataset.map(_make_labels, batched=False)

    return dataset


def _get_a_rl_dataset(
    *,
    cfg: DatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    seed: int | None = None,
) -> Dataset:
    dataset = _load_json_dataset(cfg.rl_path)

    if cfg.max_num_traces is not None:
        if cfg.trace_cap_strategy == "first":
            cap_fn = _cap_traces_per_sample
        elif cfg.trace_cap_strategy == "source_order":
            cap_fn = _cap_traces_per_sample_by_source_order
        elif cfg.trace_cap_strategy == "random":
            rng = random.Random(seed) if seed is not None else None
            cap_fn = partial(_cap_traces_per_sample_random, rng=rng)
        else:
            raise ValueError(
                "trace_cap_strategy must be one of: first, source_order, random"
            )

        dataset = dataset.map(partial(cap_fn, cap=cfg.max_num_traces))

    dataset = dataset.remove_columns(
        [
            name
            for name in ["prompt", "predictions", "sources"]
            if name in dataset.column_names
        ]
    )

    dataset = dataset.map(
        _explode_batch, batched=True, remove_columns=dataset.column_names
    )
    dataset = dataset.map(_make_completion)
    dataset = _tokenize(
        dataset=dataset, tokenizer=tokenizer, max_size=cfg.max_completion_len
    )
    keep = ["input_ids", "labels"]
    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in keep])
    return dataset


def _get_rl_datasets(
    *,
    cfgs: list[DatasetConfig],
    tokenizer: PreTrainedTokenizerBase,
    seed: int | None = None,
) -> DatasetDict:
    datasets = {}
    for idx, cfg in enumerate(cfgs):
        cfg_seed = None if seed is None else seed + idx
        datasets[cfg.name] = _get_a_rl_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=cfg_seed,
        )
        datasets[cfg.name] = select_random(
            dataset=datasets[cfg.name], n=cfg.max_num_rl, seed=cfg_seed
        )

    return DatasetDict(datasets)


def _get_fn_datasets(
    *,
    cfgs: list[DatasetConfig],
    tokenizer: PreTrainedTokenizerBase,
    n_repeats: int | None,
    seed: int | None = None,
) -> DatasetDict:
    fn_datasets = {
        cfg.name: get_fine_tuning_dataset(
            dataset_name=cfg.name, dataset_path=cfg.fn_path, tokenizer=tokenizer
        )
        for cfg in cfgs
    }
    keep = ["input_ids", "labels"]
    fn_datasets = {
        name: ds.remove_columns([x for x in ds.column_names if x not in keep])
        for name, ds in fn_datasets.items()
    }

    fn_datasets = {
        cfg.name: select_random(
            dataset=fn_datasets[cfg.name],
            n=cfg.max_num_fn,
            seed=None if seed is None else seed + idx,
        )
        for idx, cfg in enumerate(cfgs)
    }

    if n_repeats and n_repeats > 1:
        fn_datasets = {
            name: concatenate_datasets([ds for _ in range(n_repeats)])
            for name, ds in fn_datasets.items()
        }

    fn_datasets = DatasetDict(fn_datasets)

    return fn_datasets


def select_random(dataset: Dataset, n: int | None, seed: int | None = None):
    if n is None or n <= 0:
        return dataset

    dataset = dataset.shuffle(seed=seed).select(range(min(n, len(dataset))))
    return dataset


def _concat(*datasets: list[DatasetDict]) -> DatasetDict:
    names = datasets[0].keys()
    concated = {
        name: concatenate_datasets([d[name] for d in datasets]) for name in names
    }
    concated = DatasetDict(concated)
    return concated


def _apply_sample_weight(datasets: DatasetDict) -> DatasetDict:
    sizes = {name: len(ds) for name, ds in datasets.items()}
    if any(size == 0 for size in sizes.values()):
        raise ValueError("All datasets must be non-empty to compute class weights.")
    K = len(sizes)
    N = sum(list(sizes.values()))
    compute_weight = lambda n: [N / (K * n)]

    datasets = DatasetDict(
        {
            name: ds.map(lambda example: {"sample_weight": compute_weight(sizes[name])})
            for name, ds in datasets.items()
        }
    )

    return datasets


def get_dataset(
    *,
    cfgs: list[DatasetConfig],
    tokenizer: PreTrainedTokenizerBase,
    apply_class_weight: bool = False,
    n_repeats: int | None = None,
    seed: int | None = None,
):

    if cfgs[0].rl_path is None and cfgs[0].fn_path is None:
        raise ValueError("At least RL dataset or FN dataset should be provided.")

    fn_datasets = None
    if cfgs[0].fn_path is not None:
        fn_datasets = _get_fn_datasets(
            cfgs=cfgs, tokenizer=tokenizer, n_repeats=n_repeats, seed=seed
        )

    rl_datasets = None
    if cfgs[0].rl_path is not None:
        rl_datasets = _get_rl_datasets(
            cfgs=cfgs,
            tokenizer=tokenizer,
            seed=seed,
        )

    if rl_datasets is None:
        datasets = fn_datasets
    elif fn_datasets is None:
        datasets = rl_datasets
    else:
        datasets = _concat(rl_datasets, fn_datasets)

    if apply_class_weight:
        datasets = _apply_sample_weight(datasets)

    return concatenate_datasets([datasets[name] for name in datasets.keys()])


@dataclass
class DataCollatorForSupervisedFinetuning:
    pad_token_id: int
    label_pad_token_id: int = -100
    max_length: int | None = None

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not examples:
            return {}

        # Extract sequences
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        labels = [torch.tensor(e["labels"], dtype=torch.long) for e in examples]
        sample_weights = (
            None
            if "sample_weight" not in examples[0]
            else torch.tensor(
                [e["sample_weight"] for e in examples], dtype=torch.float32
            )
        )

        # Pad input_ids
        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        # Pad labels (ignore loss on padding)
        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        if self.max_length is not None:
            input_ids_padded = input_ids_padded[:, : self.max_length]
            labels_padded = labels_padded[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]

        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }

        if sample_weights is not None:
            batch["sample_weights"] = sample_weights

        return batch
