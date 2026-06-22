"""
Multi-dataset vLLM evaluator that routes each prompt to a LoRA adapter.

After preprocessing:
  - examples are grouped by adapter_path for efficient LoRA evaluation
  - outputs are then ungrouped/regrouped by dataset
  - one JSONL file is saved per dataset
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

from lora_offline.eval_dataset import get_dataset
from lora_offline.utils import get_max_length


BASE_ADAPTER_KEY = "__base__"


def _match_label(label: Any, prediction: str) -> bool:
    if prediction is None:
        return False
    label_str = "" if label is None else str(label)
    pred_str = str(prediction)
    return label_str.strip().lower() == pred_str.strip().lower()


def _safe_adapter_name(adapter_path: str) -> str:
    name = Path(adapter_path).name
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name or "adapter"


@dataclass
class DatasetSpec:
    name: str
    path: Path


@dataclass
class PromptItem:
    dataset: str
    index: int
    prompt: str
    input_ids: list[int]
    label: Any
    predicted_label: Optional[str]
    adapter_path: Optional[str]


@dataclass
class MultiVLLMEvalConfig:
    model_name: str
    datasets: list[DatasetSpec]
    routing_json_path: Path
    output_dir: Path
    max_new_tokens: int
    num_samples: int
    temperature: float
    gpu_memory_utilization: float
    swap_space: int
    cpu_offload_gb: int
    no_prefix_caching: bool
    no_eager: bool
    max_lora_rank: int
    match_label: Callable[[Any, str], bool] = _match_label


@dataclass
class GenerationRecord:
    dataset: str
    index: int
    adapter_path: Optional[str]
    predicted_adapter_label: Optional[str]
    prompt: str
    input_ids: list[int]
    label: Any
    predictions: list[str]
    prediction_token_ids: list[list[int]]
    finish_reasons: list[str | None]
    matched: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "index": self.index,
                "adapter_path": self.adapter_path,
                "predicted_adapter_label": self.predicted_adapter_label,
                "prompt": self.prompt,
                "input_ids": self.input_ids,
                "label": self.label,
                "predictions": self.predictions,
                "prediction_token_ids": self.prediction_token_ids,
                "finish_reasons": self.finish_reasons,
                "matched": self.matched,
            },
            ensure_ascii=False,
        )


class MultiVLLMEvaluator:
    def __init__(self, config: MultiVLLMEvalConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        eos_token_id = self.tokenizer.eos_token_id
        self.stop_token_ids = [eos_token_id] if eos_token_id is not None else None

        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        self.max_model_len = get_max_length(model=None, config=hf_config)

        self.routing = self._load_routing_json()
        self.llm: LLM | None = None
        self.adapter_to_lora_id: dict[str, int] = {}

    def run(self) -> None:
        prompt_items = self._load_and_preprocess_all_datasets()
        grouped_items = self._group_by_adapter(prompt_items)

        self._ensure_output_dir()
        self._init_llm(grouped_items)

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
            stop_token_ids=self.stop_token_ids,
            detokenize=False,
            n=self.config.num_samples,
        )

        all_records: list[GenerationRecord] = []

        for adapter_key, items in grouped_items.items():
            adapter_path = None if adapter_key == BASE_ADAPTER_KEY else adapter_key
            print(
                f"Running group: adapter={adapter_path or 'BASE MODEL'} "
                f"num_prompts={len(items)}"
            )

            records = self._generate_for_group(
                items=items,
                adapter_path=adapter_path,
                sampling_params=sampling_params,
            )
            all_records.extend(records)

        self._write_results_by_dataset(all_records)
        self._report_accuracy(all_records)

    def _load_routing_json(self) -> dict[str, dict[str, Any]]:
        with self.config.routing_json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _ensure_output_dir(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_preprocess_all_datasets(self) -> list[PromptItem]:
        items: list[PromptItem] = []

        for dataset_spec in self.config.datasets:
            dataset = get_dataset(
                dataset_name=dataset_spec.name,
                dataset_path=str(dataset_spec.path),
                tokenizer=self.tokenizer,
                max_length=self.max_model_len,
            )

            dataset_routing = self.routing.get(dataset_spec.name, {})

            for idx, example in enumerate(dataset):
                route = dataset_routing.get(str(idx))

                predicted_label: Optional[str] = None
                adapter_path: Optional[str] = None

                if isinstance(route, dict):
                    predicted_label = route.get("predicted_label")
                    adapter_path = route.get("adapter_path")
                elif isinstance(route, str):
                    predicted_label = route

                items.append(
                    PromptItem(
                        dataset=dataset_spec.name,
                        index=idx,
                        prompt=example["prompt"],
                        input_ids=example["input_ids"],
                        label=example["label"],
                        predicted_label=predicted_label,
                        adapter_path=adapter_path,
                    )
                )

        return items

    def _group_by_adapter(
        self,
        items: list[PromptItem],
    ) -> dict[str, list[PromptItem]]:
        grouped: dict[str, list[PromptItem]] = {}

        for item in items:
            adapter_key = item.adapter_path or BASE_ADAPTER_KEY
            grouped.setdefault(adapter_key, []).append(item)

        return grouped

    def _init_llm(self, grouped_items: dict[str, list[PromptItem]]) -> None:
        if self.llm is not None:
            return

        adapter_paths = [
            adapter_key
            for adapter_key in grouped_items
            if adapter_key != BASE_ADAPTER_KEY
        ]
        use_lora = len(adapter_paths) > 0

        self.adapter_to_lora_id = {
            adapter_path: i + 1
            for i, adapter_path in enumerate(sorted(adapter_paths))
        }

        self.llm = LLM(
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            enable_lora=use_lora,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            swap_space=self.config.swap_space,
            cpu_offload_gb=self.config.cpu_offload_gb,
            enforce_eager=not self.config.no_eager,
            enable_prefix_caching=not self.config.no_prefix_caching,
            max_lora_rank=self.config.max_lora_rank if use_lora else None,
        )

        if not use_lora:
            print("⚠️ No LoRA adapters found in routing JSON — running base model only.")

    def _build_lora_request(
        self,
        adapter_path: Optional[str],
    ) -> Optional[LoRARequest]:
        if adapter_path is None:
            return None

        return LoRARequest(
            lora_name=_safe_adapter_name(adapter_path),
            lora_int_id=self.adapter_to_lora_id[adapter_path],
            lora_path=adapter_path,
            base_model_name=self.config.model_name,
        )

    def _generate_for_group(
        self,
        items: list[PromptItem],
        adapter_path: Optional[str],
        sampling_params: SamplingParams,
    ) -> list[GenerationRecord]:
        assert self.llm is not None

        prompts = [
            TokensPrompt(prompt_token_ids=item.input_ids)
            for item in items
        ]

        lora_request = self._build_lora_request(adapter_path)

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
        )

        records: list[GenerationRecord] = []

        for output, item in zip(outputs, items):
            token_seqs = [list(candidate.token_ids) for candidate in output.outputs]
            predictions = [
                self.tokenizer.decode(token_ids, skip_special_tokens=True)
                for token_ids in token_seqs
            ]
            finish_reasons = [
                candidate.finish_reason
                for candidate in output.outputs
            ]

            first_prediction = predictions[0] if predictions else ""
            matched = self.config.match_label(item.label, first_prediction)

            records.append(
                GenerationRecord(
                    dataset=item.dataset,
                    index=item.index,
                    adapter_path=adapter_path,
                    predicted_adapter_label=item.predicted_label,
                    prompt=item.prompt,
                    input_ids=item.input_ids,
                    label=item.label,
                    predictions=predictions,
                    prediction_token_ids=token_seqs,
                    finish_reasons=finish_reasons,
                    matched=matched,
                )
            )

        return records

    def _ungroup_by_dataset(
        self,
        records: list[GenerationRecord],
    ) -> dict[str, list[GenerationRecord]]:
        records_by_dataset: dict[str, list[GenerationRecord]] = {}

        for record in records:
            records_by_dataset.setdefault(record.dataset, []).append(record)

        for dataset_name in records_by_dataset:
            records_by_dataset[dataset_name].sort(key=lambda r: r.index)

        return records_by_dataset

    def _write_results_by_dataset(self, records: list[GenerationRecord]) -> None:
        records_by_dataset = self._ungroup_by_dataset(records)

        for dataset_name, dataset_records in sorted(records_by_dataset.items()):
            output_path = self.config.output_dir / f"{dataset_name}.jsonl"

            with output_path.open("w", encoding="utf-8") as handle:
                for record in dataset_records:
                    handle.write(record.to_json() + "\n")

            print(f"Wrote {len(dataset_records)} records to {output_path}")

    def _report_accuracy(self, records: list[GenerationRecord]) -> None:
        if not records:
            print("No records generated.")
            return

        total = len(records)
        correct = sum(1 for record in records if record.matched)
        accuracy = correct / total

        print(f"Overall accuracy: {correct}/{total} = {accuracy:.2%}")

        records_by_dataset = self._ungroup_by_dataset(records)

        for dataset_name, dataset_records in sorted(records_by_dataset.items()):
            dataset_total = len(dataset_records)
            dataset_correct = sum(1 for record in dataset_records if record.matched)
            dataset_accuracy = dataset_correct / dataset_total if dataset_total else 0.0
            print(
                f"{dataset_name}: "
                f"{dataset_correct}/{dataset_total} = {dataset_accuracy:.2%}"
            )


def parse_dataset_specs(values: list[str]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []

    for value in values:
        if "=" not in value:
            raise ValueError(
                "Each --dataset must have format dataset_name=/path/to/file.jsonl. "
                f"Got: {value}"
            )

        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()

        if not name:
            raise ValueError(f"Dataset name is empty in --dataset {value!r}")
        if not path:
            raise ValueError(f"Dataset path is empty in --dataset {value!r}")

        specs.append(DatasetSpec(name=name, path=Path(path).expanduser()))

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple datasets with vLLM, route prompts to LoRA adapters, "
            "and save one JSONL per dataset."
        )
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model name or path.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset specification in the form dataset_name=/path/to/file.jsonl. "
            "Pass this argument multiple times for multiple datasets."
        ),
    )
    parser.add_argument(
        "--routing-json",
        required=True,
        type=Path,
        help="Path to JSON produced by extract_predictions.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where per-dataset JSONL files will be written.",
    )
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=int, default=20)
    parser.add_argument("--cpu-offload-gb", type=int, default=0)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--no-eager", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    config = MultiVLLMEvalConfig(
        model_name=args.model_name,
        datasets=parse_dataset_specs(args.dataset),
        routing_json_path=args.routing_json.expanduser(),
        output_dir=args.output_dir.expanduser(),
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        cpu_offload_gb=args.cpu_offload_gb,
        no_prefix_caching=args.no_prefix_caching,
        no_eager=args.no_eager,
        max_lora_rank=args.max_lora_rank,
    )

    evaluator = MultiVLLMEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()