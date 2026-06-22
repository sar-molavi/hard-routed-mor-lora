"""
Simple object-oriented vLLM evaluator for LoRA adapters over a single dataset.

The script loads an evaluation dataset via ``lora.eval_dataset``, runs generation
with a vLLM engine that has the requested LoRA adapter applied, and writes JSONL
results containing inputs, labels, decoded predictions, and their generated
token IDs.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

from .eval_dataset import get_dataset
from .utils import get_max_length


def _match_label(label: Any, prediction: str) -> bool:
    """
    Compare the ground-truth label with the model prediction using a case-insensitive,
    whitespace-trimmed exact match. Handles non-string labels by casting to ``str``.
    """
    if prediction is None:
        return False

    label_str = "" if label is None else str(label)
    pred_str = str(prediction)
    return label_str.strip().lower() == pred_str.strip().lower()


@dataclass
class VLLMEvalConfig:
    """Container for all evaluator runtime settings."""

    model_name: str
    lora_path: Path
    adapter_name: str
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
    enable_thinking: bool = False
    match_label: Callable[[Any, str], bool] = _match_label


@dataclass
class GenerationRecord:
    """Single evaluation record containing decoded outputs and raw token IDs."""

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
        """Serialize the record as a JSON string ready to be written to disk."""
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
    """Orchestrates dataset loading, vLLM inference, and result scoring."""

    def __init__(self, config: VLLMEvalConfig) -> None:
        """Prepare the evaluator with the supplied configuration object."""
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        eos_token_id = self.tokenizer.eos_token_id
        self.stop_token_ids = [eos_token_id] if eos_token_id is not None else None

        hf_config = AutoConfig.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.max_model_len = get_max_length(model=None, config=hf_config)

        self.llm: LLM | None = None
        self.lora_request: LoRARequest | None = None

    def run(self) -> None:
        """
        Execute the full evaluation pipeline: load the dataset, run vLLM inference,
        persist generations, and report aggregate accuracy.
        """
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
        """Create the output directory tree if it does not already exist."""
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_llm(self) -> None:
        """Instantiate the vLLM engine and build the LoRA request handle."""
        if self.llm is not None:
            return

        self.llm = LLM(
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            enable_lora=True,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            swap_space=self.config.swap_space,
            cpu_offload_gb=self.config.cpu_offload_gb,
            enforce_eager=not self.config.no_eager,
            enable_prefix_caching=not self.config.no_prefix_caching,
            max_lora_rank=self.config.max_lora_rank,
        )
        self.lora_request = LoRARequest(
            lora_name=self.config.adapter_name,
            lora_int_id=1,
            lora_path=str(self.config.lora_path),
            base_model_name=self.config.model_name,
        )

    def _load_dataset(self):
        """Materialize the eval dataset using the shared tokenizer and max length."""
        return get_dataset(
            dataset_name=self.config.dataset_name,
            dataset_path=str(self.config.dataset_path),
            tokenizer=self.tokenizer,
            max_length=self.max_model_len,
            enable_thinking=self.config.enable_thinking,
        )

    def _generate_for_dataset(
        self,
        dataset_name: str,
        dataset,
        sampling_params: SamplingParams,
    ) -> list[GenerationRecord]:
        """
        Generate completions for all examples in a dataset and collect result records.

        Args:
            dataset_name: Logical name of the dataset (e.g., "math", "cola").
            dataset: Iterable of tokenized prompt examples returned by ``get_dataset``.
            sampling_params: vLLM sampling configuration shared across prompts.

        Returns:
            A list of ``GenerationRecord`` objects capturing predictions and labels.
        """
        assert self.llm is not None
        assert self.lora_request is not None

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
            lora_request=self.lora_request,
        )
        records: list[GenerationRecord] = []

        for output, (idx, meta) in zip(outputs, metadata):
            token_seqs = [list(completion.token_ids) for completion in output.outputs]
            finish_reasons = [completion.finish_reason for completion in output.outputs]
            predictions = [
                self.tokenizer.decode(tokens, skip_special_tokens=True)
                for tokens in token_seqs
            ]
            matched = self.config.match_label(
                meta["label"], predictions[0] if predictions else ""
            )
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
        """Write every generation record to the JSONL output file."""
        with self.config.output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json() + "\n")

    def _report_accuracy(self, records: list[GenerationRecord]) -> None:
        """Print a simple exact-match accuracy summary for the generated results."""
        if not records:
            print("No records generated.")
            return

        total = len(records)
        correct = sum(1 for record in records if record.matched)
        accuracy = correct / total
        print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")


def parse_args() -> argparse.Namespace:
    """Declare and parse CLI arguments for running the evaluator."""
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters with vLLM.")
    parser.add_argument("--model-name", required=True, help="Base model name or path.")
    parser.add_argument(
        "--lora-path",
        required=True,
        type=Path,
        help="Path to the directory containing the LoRA adapter.",
    )
    parser.add_argument(
        "--adapter-name",
        default="default",
        help="Adapter identifier passed to vLLM when loading the LoRA.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Logical name for the dataset (used in reports and output).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the JSONL evaluation results.",
    )
    parser.add_argument(
        "--max-lora-rank", type=int, default=64, help="The maximum LoRA rank to use."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per request.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completions to request per prompt.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature forwarded to vLLM (default: 0.0).",
    )

    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional CUDA device mask (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM can use (default: 0.90).",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=20,
        help="Host memory (GB) reserved for swap (default: 20).",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=0,
        help="CPU offload memory (GB) allocated by vLLM (default: 0).",
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable vLLM KV prefix caching when set.",
    )
    parser.add_argument(
        "--no-eager",
        action="store_true",
        help="Disable enforced eager execution (allow CUDA graphs).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in tokenizer.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point: parse arguments, build the evaluator, and launch it."""
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    config = VLLMEvalConfig(
        model_name=args.model_name,
        lora_path=args.lora_path.expanduser(),
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
        enable_thinking=args.enable_thinking,
    )
    evaluator = VLLMEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
