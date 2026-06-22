"""Prompt-focused dataset helpers for TRL GRPO fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping
import json

from transformers import PreTrainedTokenizer
from datasets import Dataset, concatenate_datasets

from . import prompts as prompt_templates
from .utils import extract_last_after_hashes


Formatter = Callable[[Dataset], Dataset]


def _choices_to_str(choices: Mapping[str, str | None]) -> str:
    """Render multiple-choice options as a labeled block of text."""
    return "\n\n".join(f"{key}) {value}" for key, value in choices.items() if value)


def _make_prompt(
    example: Mapping[str, object], *, prompt_template: str
) -> dict[str, object]:
    """Format the example into the prompt structure required downstream."""
    options = _choices_to_str(example["options"])
    prompt = prompt_template.format(options=options, question=example["question"])
    return {"prompt": prompt, "ground_truth": example["answer_idx"].lower().strip()}


def _prepare_medical_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.Medical_PROMPT
    dataset = dataset.map(
        lambda record: _make_prompt(record, prompt_template=template),
        batched=False,
        desc="Formatting medical prompts",
    )
    return dataset


def _format_arc_choices(choices: dict[str, list[str]]) -> str:
    """Render ARC multiple-choice options as newline separated strings."""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def _prepare_svamp_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.MATH

    def build_prompt(example: dict) -> dict[str, str]:
        promot = example["Body"] + "\n" + example["Question"]
        return {
            "prompt": template.format(question=promot),
            "ground_truth": (example["Answer"]).strip().lower(),
        }

    return dataset.map(build_prompt, desc="Formatting SVAMP prompts")


def _prepare_math_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.MATH

    def build_prompt(example: dict) -> dict[str, str]:
        return {
            "prompt": template.format(question=example["question"]),
            "ground_truth": extract_last_after_hashes(example["answer"])
            .strip()
            .lower(),
        }

    return dataset.map(build_prompt, desc="Formatting math prompts")


def _prepare_coding_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.CODING
    if not template:
        raise ValueError(
            "CODING prompt template is empty. Please update `lora_trl/prompts.py`."
        )

    def build_prompt(example: dict) -> dict[str, str]:
        return {
            "prompt": template.format(text=example["prompt"]),
            "ground_truth": example["canonical_solution"],
        }

    return dataset.map(build_prompt, desc="Formatting coding prompts")


def _prepare_sst2_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.SST2

    def build_prompt(example: dict) -> dict[str, str]:
        return {
            "prompt": template.format(text=example["sentence"]),
            "ground_truth": str(example["label"]),
        }

    return dataset.map(build_prompt, desc="Formatting SST-2 prompts")


def _prepare_cola_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.COLA

    def build_prompt(example: dict) -> dict[str, str]:
        return {
            "prompt": template.format(text=example["sentence"]),
            "ground_truth": str(example["label"]),
        }

    return dataset.map(build_prompt, desc="Formatting CoLA prompts")


def _prepare_sst2_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.SST2

    def build_prompt(example: dict) -> dict[str, str]:
        return {
            "prompt": template.format(text=example["sentence"]),
            "ground_truth": str(example["label"]),
        }

    return dataset.map(build_prompt, desc="Formatting SST prompts")


def _prepare_arc_dataset(dataset: Dataset) -> Dataset:
    template = prompt_templates.ARC

    def build_prompt(example: dict) -> dict[str, str]:
        choices_text = _format_arc_choices(example["choices"])
        return {
            "prompt": template.format(
                text=example["question"], choices_text=choices_text
            ),
            "ground_truth": example["answerKey"],
        }

    return dataset.map(build_prompt, desc="Formatting ARC prompts")


def _prepare_boolq_dataset(dataset: Dataset, reasoning=True) -> Dataset:
    template = prompt_templates.BOOLQ

    def build_prompt(example: dict) -> dict[str, str]:
        answer = example["answer"]

        return {
            "prompt": template.format(
                question=example["question"], passage=example["passage"]
            ),
            "ground_truth": str(int(example["answer"])),
        }

    return dataset.map(build_prompt, desc="Formatting boolq prompts")


FORMATTERS: dict[str, Formatter] = {
    "math": _prepare_math_dataset,
    "coding": _prepare_coding_dataset,
    "sst": _prepare_sst2_dataset,
    "cola": _prepare_cola_dataset,
    "arc": _prepare_arc_dataset,
    "medical": _prepare_medical_dataset,
    "boolq": _prepare_boolq_dataset,
    "svamp": _prepare_svamp_dataset,
    "sst": _prepare_sst2_dataset,
}


def _read_json_using_json(path: str | Path) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def _load_json_dataset(path: Path) -> Dataset:
    """Load a JSONL dataset from a single file or concatenate multiple splits."""
    if path.is_file():
        return _read_json_using_json(path)

    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found in directory: {path}")
        datasets = [Dataset.from_json(str(f)) for f in files]
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def _apply_chat_template(dataset: Dataset, tokenizer: PreTrainedTokenizer):
    def _message_from_prompt(x):
        return {"prompt": [{"role": "user", "content": x["prompt"]}]}

    def _apply(x):
        return {
            "prompt": tokenizer.apply_chat_template(
                x["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
        }

    dataset = dataset.map(_message_from_prompt)
    dataset = dataset.map(_apply, batched=True)
    return dataset


def _apply_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:

    def _apply_format(example: dict[str, str]) -> dict[str, str]:
        return {
            "prompt": tokenizer.bos_token + example["prompt"] + "\n\n---\n\nAnswer:\n"
        }

    dataset = dataset.map(_apply_format, desc="Applying template", batched=False)
    return dataset


def get_dataset(
    *, dataset_name: str, dataset_path: str, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """
    Load a raw JSONL dataset and attach RL-ready prompt/ground-truth columns.
    """
    name = dataset_name.lower()
    if name not in FORMATTERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Available options: {', '.join(sorted(FORMATTERS))}"
        )

    dataset = _load_json_dataset(Path(dataset_path))
    dataset = FORMATTERS[name](dataset)
    if getattr(tokenizer, "chat_template", None) is None:
        dataset = _apply_template(dataset=dataset, tokenizer=tokenizer)
    else:
        dataset = _apply_chat_template(
            dataset=dataset,
            tokenizer=tokenizer,
        )

    required_columns = {"prompt", "ground_truth"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise RuntimeError(
            f"Dataset '{dataset_name}' is missing columns: {sorted(missing)}"
        )

    return dataset
