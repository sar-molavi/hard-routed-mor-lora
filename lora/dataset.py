from __future__ import annotations

from pathlib import Path
from typing import Callable
from dataclasses import dataclass
import json
import random
from functools import partial

from datasets import Dataset, Value, concatenate_datasets
from transformers import PreTrainedTokenizerBase
import torch
from torch.nn.utils.rnn import pad_sequence

from lora_offline import prompts as prompt_templates
from .utils import _extract_gsm8k_answer, _extract_gsm8k_reasoning


Formatter = Callable[[Dataset], Dataset]

LABEL_TEMPLATE = """
<think>
{reasoning}
</think>

{{
  "answer": "{answer}"
}}
""".strip()


def _garmmar_label(label: int | str) -> str:
    label = str(label).strip()
    R = "acceptable" if label == "1" else "unacceptable"
    completion = LABEL_TEMPLATE.format(
        reasoning=f"The sentence is evaluated as grammatically {R}.", answer=label
    )

    return {
        "completion": completion,
        "label": label,
    }


def _arc_label(label: int | str) -> str:
    label = str(label).strip()
    completion = LABEL_TEMPLATE.format(
        reasoning=f"have selected option {label} as the correct answer.", answer=label
    )

    return {
        "completion": completion,
        "label": label,
    }


def _medqa_label(label: int | str) -> str:
    label = str(label).strip()
    completion = LABEL_TEMPLATE.format(
        reasoning=f"have selected option {label} as the best answer.", answer=label
    )

    return {
        "completion": completion,
        "label": label,
    }


def _gsm8k_label_no_reasoning(label: int | str) -> str:
    answer = label.strip()
    label = _extract_gsm8k_answer(answer).strip()

    completion = LABEL_TEMPLATE.format(
        reasoning=f"I translated the word problem into a computation and determined the final result. The answer is {label}",
        answer=label,
    )

    return {
        "completion": completion,
        "label": label,
    }


def _gsm8k_label(label: int | str) -> str:
    answer = label.strip()
    reasoning = _extract_gsm8k_reasoning(answer).strip()
    label = _extract_gsm8k_answer(answer).strip()

    completion = LABEL_TEMPLATE.format(reasoning=reasoning, answer=label)

    return {
        "completion": completion,
        "label": label,
    }


def _boolq_label(label: int | str | bool) -> str:
    answer = int(bool(label))
    completion = LABEL_TEMPLATE.format(
        reasoning=f"Based on the given passage, it is {"true" if bool(label) else "false"}.",
        answer=str(answer),
    )
    return {
        "completion": completion,
        "label": answer,
    }


def _format_arc_choices(choices: dict[str, list[str]]) -> str:
    """Render ARC multiple-choice options as newline separated strings."""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def _prepare_math_dataset(dataset: Dataset, reasoning=True) -> Dataset:
    template = prompt_templates.MATH

    def build_prompt(example: dict) -> dict[str, str]:
        answer = example["answer"]

        return {
            "prompt": template.format(question=example["question"]),
            **(
                _gsm8k_label(answer) if reasoning else _gsm8k_label_no_reasoning(answer)
            ),
        }

    return dataset.map(build_prompt, desc="Formatting math prompts")


def _prepare_cola_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.COLA

    def build_prompt(example: dict) -> dict[str, str]:
        return {
            "prompt": template.format(text=example["sentence"]),
            **_garmmar_label(example["label"]),
        }

    return dataset.map(build_prompt, desc="Formatting CoLA prompts")


def _prepare_arc_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.ARC

    def build_prompt(example: dict) -> dict[str, str]:
        choices_text = _format_arc_choices(example["choices"])
        return {
            "prompt": template.format(
                text=example["question"], choices_text=choices_text
            ),
            **_arc_label(example["answerKey"]),
        }

    return dataset.map(build_prompt, desc="Formatting ARC prompts")


def _prepare_medqa_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.Medical_PROMPT

    def _choices_to_str(choices: dict[str, str | None]) -> str:
        """Render multiple-choice options as a labeled block of text."""
        return "\n".join(f"{key}) {value}" for key, value in choices.items() if value)

    def build_prompt(example: dict) -> dict[str, str]:
        options = _choices_to_str(example["options"])
        prompt = template.format(options=options, question=example["question"])
        return {
            "prompt": prompt,
            **_medqa_label(example["answer_idx"]),
        }

    return dataset.map(build_prompt, desc="Formatting MedQA prompts")


def _prepare_boolq_dataset(dataset: Dataset, reasoning=True) -> Dataset:
    template = prompt_templates.BOOLQ

    def build_prompt(example: dict) -> dict[str, str]:
        answer = example["answer"]

        return {
            "prompt": template.format(
                question=example["question"], passage=example["passage"]
            ),
            **_boolq_label(answer),
        }

    return dataset.map(build_prompt, desc="Formatting boolq prompts")


FORMATTERS: dict[str, Formatter] = {
    "math": _prepare_math_dataset,
    "math_no_reasoning": partial(_prepare_math_dataset, reasoning=False),
    "cola": _prepare_cola_dataset,
    "arc": _prepare_arc_dataset,
    "medqa": _prepare_medqa_dataset,
    "boolq": _prepare_boolq_dataset,
}


def _read_json_using_json(path: str | Path) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def _load_json_dataset(path: Path) -> Dataset:
    """Load a JSONL dataset from a single file."""
    if path.is_file():
        return _read_json_using_json(path)

    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def _apply_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:

    def _apply_format(example: dict[str, str]) -> dict[str, str]:
        return {"prompt": example["prompt"] + "\n\n---\n\nAnswer:\n"}

    def _apply_simple_template(example: dict[str, str]) -> dict[str, str]:
        results = {}

        results["prompt_ids"] = tokenizer(
            example["prompt"], add_special_tokens=True, return_attention_mask=False
        )["input_ids"]

        results["completion_ids"] = tokenizer(
            example["completion"], add_special_tokens=False, return_attention_mask=False
        )["input_ids"]

        results["completion_ids"] = [
            x + [tokenizer.eos_token_id] for x in results["completion_ids"]
        ]

        return results

    def _make_input_ids(example: dict[str, list[int]]) -> dict[str, int]:
        input_ids = example["prompt_ids"].copy()
        input_ids += example["completion_ids"]

        labels = input_ids.copy()
        for i in range(len(example["prompt_ids"])):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(_apply_format, desc="Applying the format", batched=False)

    dataset = dataset.map(
        _apply_simple_template, desc="Applying chat template", batched=True
    )

    dataset = dataset.map(_make_input_ids, batched=False, desc="Make input ids")

    return dataset


def _apply_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Wrap plain-text prompts with the tokenizer's chat template.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise ValueError(
            "Tokenizer must implement `apply_chat_template` to format chat prompts."
        )

    def _format_prompt(example: dict[str, str]) -> dict[str, str]:
        prompt = [{"role": "user", "content": example["prompt"]}]
        return {"prompt": prompt}

    def _apply_template(example: dict[str, str]) -> dict[str, str]:
        results = {}

        results["prompt_ids"] = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )

        results["completion_ids"] = tokenizer(
            example["completion"],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        return results

    def _make_input_ids(example: dict[str, list[int]]) -> dict[str, int]:
        input_ids = example["prompt_ids"].copy()
        input_ids += example["completion_ids"] + [tokenizer.eos_token_id]

        labels = input_ids.copy()
        for i in range(len(example["prompt_ids"])):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(
        _format_prompt,
        desc="Formatting chat prompts",
    )

    dataset = dataset.map(
        _apply_template,
        batched=True,
        desc="Applying chat template",
    )

    dataset = dataset.map(
        _make_input_ids,
        batched=False,
        desc="Make input ids",
    )

    return dataset


def get_dataset(
    *,
    dataset_name: str | list[str],
    dataset_path: str | list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_training_sample: int | None = None,
    seed: int | None = None,
) -> Dataset:
    """
    Load one or more raw JSONL datasets and attach SFT-ready columns.
    """
    dataset_names = [dataset_name] if isinstance(dataset_name, str) else dataset_name
    dataset_paths = [dataset_path] if isinstance(dataset_path, str) else dataset_path

    if not dataset_names:
        raise ValueError("`dataset_name` must not be empty.")
    if not dataset_paths:
        raise ValueError("`dataset_path` must not be empty.")
    if len(dataset_names) == 1 and len(dataset_paths) > 1:
        dataset_names = dataset_names * len(dataset_paths)
    elif len(dataset_paths) == 1 and len(dataset_names) > 1:
        dataset_paths = dataset_paths * len(dataset_names)
    elif len(dataset_names) != len(dataset_paths):
        raise ValueError(
            "`dataset_name` and `dataset_path` must have the same length. "
            f"Got {len(dataset_names)} and {len(dataset_paths)}."
        )

    datasets = []
    rng = random.Random(seed) if seed is not None else None

    for name_raw, path_raw in zip(dataset_names, dataset_paths):
        name = name_raw.lower()
        if name not in FORMATTERS:
            raise ValueError(
                f"Unsupported dataset '{name_raw}'. "
                f"Available options: {', '.join(sorted(FORMATTERS))}"
            )

        ds = _load_json_dataset(Path(path_raw))
        ds = FORMATTERS[name](ds)
        ds = ds.remove_columns(
            [col for col in ds.column_names if col not in {"prompt", "completion", "label"}]
        )

        # Keep label dtype consistent across mixed datasets (e.g., boolq -> int, arc -> str).
        if "label" in ds.column_names:
            ds = ds.cast_column("label", Value("string"))

        if max_training_sample is not None:
            if max_training_sample <= 0:
                raise ValueError("`max_training_sample` must be a positive integer.")
            if len(ds) > max_training_sample:
                indices = list(range(len(ds)))
                if rng is not None:
                    rng.shuffle(indices)
                else:
                    random.shuffle(indices)
                ds = ds.select(indices[:max_training_sample])

        datasets.append(ds)

    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    required_columns = {"prompt", "completion", "label"}
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in required_columns]
    )

    if getattr(tokenizer, "chat_template", None) is None:
        dataset = _apply_template(dataset=dataset, tokenizer=tokenizer)
    else:
        dataset = _apply_chat_template(
            dataset=dataset,
            tokenizer=tokenizer,
        )

    return dataset


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

        return batch


__all__ = ["get_dataset", "DataCollatorForSupervisedFinetuning"]
