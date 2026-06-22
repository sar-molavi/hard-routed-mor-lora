"""
Simple object-oriented vLLM evaluator for LoRA adapters over a single dataset.

If --lora-path is provided, the LoRA adapter will be applied.
If --lora-path is omitted, the base model will be used without LoRA.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
  
from .eval_dataset import get_dataset
from .utils import get_max_length


def _match_label(label: Any, prediction: str) -> bool:
    if prediction is None:
        return False
    label_str = "" if label is None else str(label)
    pred_str = str(prediction)
    return label_str.strip().lower() == pred_str.strip().lower()


@dataclass
class VLLMEvalConfig:
    model_name: str
    lora_path: Optional[Path]           # OPTIONAL NOW
    adapter_name: Optional[str]         # OPTIONAL NOW
    dataset_name: str
    dataset_path: Path
    output_path: Path
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


class VLLMEvaluator:
    def __init__(self, config: VLLMEvalConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        eos_token_id = self.tokenizer.eos_token_id
        self.stop_token_ids = [eos_token_id] if eos_token_id is not None else None

        hf_config = AutoConfig.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.max_model_len = get_max_length(model=None, config=hf_config)

        self.llm: LLM | None = None
        self.lora_request: Optional[LoRARequest] = None

    def run(self) -> None:
        dataset = self._load_dataset()
        self._ensure_output_path()
        self._init_llm()

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
            stop_token_ids=self.stop_token_ids,
            detokenize=False,
            n=self.config.num_samples,
        )

        records = self._generate_for_dataset(
            dataset_name=self.config.dataset_name,
            dataset=dataset,
            sampling_params=sampling_params,
        )

        self._write_results(records)
        self._report_accuracy(records)

    def _ensure_output_path(self) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # INIT LLM WITH/ WITHOUT LoRA
    # -------------------------------
    def _init_llm(self) -> None:
        if self.llm is not None:
            return

        use_lora = self.config.lora_path is not None

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

        if use_lora:
            self.lora_request = LoRARequest(
                lora_name=self.config.adapter_name or "default",
                lora_int_id=1,
                lora_path=str(self.config.lora_path),
                base_model_name=self.config.model_name,
            )
        else:
            self.lora_request = None
            print("⚠️ No LoRA provided — running base model only.")

    def _load_dataset(self):
        return get_dataset(
            dataset_name=self.config.dataset_name,
            dataset_path=str(self.config.dataset_path),
            tokenizer=self.tokenizer,
            max_length=self.max_model_len,
        )

    def _generate_for_dataset(
        self, dataset_name: str, dataset, sampling_params: SamplingParams
    ) -> list[GenerationRecord]:

        assert self.llm is not None

        prompts: list[TokensPrompt] = []
        metadata: list[tuple[int, dict]] = []

        for idx, example in enumerate(dataset):
            prompts.append(TokensPrompt(prompt_token_ids=example["input_ids"]))
            metadata.append(
                (
                    idx,
                    {
                        "prompt": example["prompt"],
                        "input_ids": example["input_ids"],
                        "label": example["label"],
                    },
                )
            )

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            use_tqdm=True,
            lora_request=self.lora_request,  # OK if None
        )

        records: list[GenerationRecord] = []
        for output, (idx, meta) in zip(outputs, metadata):
            token_seqs = [list(c.token_ids) for c in output.outputs]
            predictions = [
                self.tokenizer.decode(toks, skip_special_tokens=True)
                for toks in token_seqs
            ]
            finish_reasons = [c.finish_reason for c in output.outputs]
            matched = self.config.match_label(meta["label"], predictions[0] if predictions else "")

            records.append(
                GenerationRecord(
                    dataset=dataset_name,
                    index=idx,
                    prompt=meta["prompt"],
                    input_ids=meta["input_ids"],
                    label=meta["label"],
                    predictions=predictions,
                    prediction_token_ids=token_seqs,
                    finish_reasons=finish_reasons,
                    matched=matched,
                )
            )
        return records

    def _write_results(self, records: list[GenerationRecord]) -> None:
        with self.config.output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json() + "\n")

    def _report_accuracy(self, records: list[GenerationRecord]) -> None:
        if not records:
            print("No records generated.")
            return
        total = len(records)
        correct = sum(1 for record in records if record.matched)
        accuracy = correct / total
        print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters with vLLM.")
    parser.add_argument("--model-name", required=True, help="Base model name or path.")
    parser.add_argument("--lora-path", type=Path, default=None, help="Optional LoRA path.")
    parser.add_argument("--adapter-name", default=None, help="Optional adapter identifier.")
    parser.add_argument("--dataset-name", required=True, help="Logical dataset name.")
    parser.add_argument("--dataset-path", required=True, type=Path, help="Path to JSONL dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write JSONL results.")
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

    config = VLLMEvalConfig(
        model_name=args.model_name,
        lora_path=args.lora_path.expanduser() if args.lora_path else None,
        adapter_name=args.adapter_name,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path.expanduser(),
        output_path=args.output.expanduser(),
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

    evaluator = VLLMEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
