"""Single-GPU dynamic batching evaluator for LoRA-Mixer using Hugging Face generation."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CompileConfig,
    PreTrainedTokenizerBase,
    StaticCache,
)

from lora_offline.eval_dataset import (
    get_dataset,
)  # A dataset containing two columns: tokenized prompt 'input_ids' and the ground_truth 'label'

from .config import MixerTrainingConfig
from .utils import get_max_length, load_checkpoint_path, install_exit_handlers

from .mixer import LoRAMixerFFN
from .utils import load_lora_mixer_weights, resolve_dtype


@dataclass
class EvalConfig:
    """Container for user-specified runtime options."""

    model_name_or_path: str
    lora_path: Path
    dataset_name: str
    dataset_path: Path
    output_path: Path
    max_new_tokens: int
    num_samples: int
    temperature: float
    top_p: float | None
    top_k: int | None
    router_top_k: int
    normalize_router_weights: bool
    router_alpha: float
    router_token_gamma: float
    router_sequence_gamma: float
    jitter_noise: float | None
    apply_hard: bool | None
    router_shared_across_layers: bool
    repetition_penalty: float
    device: str
    fp16: bool
    bf16: bool
    pad_to_multiple_of: int | None
    max_batch_tokens: int
    max_batch_size: int
    chunk_size: int
    expert_paths: list[Path]
    enable_lora_attn: bool
    lora_kwargs: dict[str, Any] | None
    num_layers: int | None
    max_length: int | None
    resume: bool
    enable_compile: bool


@dataclass
class GenerationRecord:
    """Represents the minimal metadata saved for each generated completion."""

    dataset: str
    prompt: str
    input_ids: list[int]
    label: Any
    prediction_token_ids: list[list[int]]
    predictions: list[str]
    finish_reason: list[str]
    index: int

    def to_json(self) -> str:
        """Serialize the record to a JSON string."""
        return json.dumps(
            {
                "dataset": self.dataset,
                "prompt": self.prompt,
                "input_ids": self.input_ids,
                "label": self.label,
                "prediction_token_ids": self.prediction_token_ids,
                "predictions": self.predictions,
                "finish_reason": self.finish_reason,
                "index": self.index,
            },
            ensure_ascii=False,
        )


class LeftPaddingCollator:
    """Pads variable-length prompts on the left while tracking original lengths."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer.padding_side = "left"

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Pad a batch of examples and return tensors plus metadata."""
        if not examples:
            return {}

        prompt_lengths = torch.tensor(
            [len(example["input_ids"]) for example in examples],
            dtype=torch.long,
        )

        batch = self.tokenizer.pad(
            {"input_ids": [example["input_ids"] for example in examples]},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "prompt_lengths": prompt_lengths,
            "prompts": [example["prompt"] for example in examples],
            "labels": [example["label"] for example in examples],
            "raw_input_ids": [example["input_ids"] for example in examples],
        }


class DynamicBatchScheduler:
    """Groups prompts into token-balanced batches suitable for generation."""

    def __init__(
        self,
        dataset,
        sorted_indices: Sequence[int],
        *,
        max_batch_size: int,
        max_batch_tokens: int,
    ) -> None:
        # Save references to the dataset and parameters controlling batch limits.
        self.dataset = (
            dataset  # The dataset containing examples (each a dict with "input_ids")
        )
        self.sorted_indices = list(
            sorted_indices
        )  # Order in which to read examples (usually sorted by length)
        self.max_batch_size = (
            max_batch_size  # Max number of examples allowed in a batch
        )
        self.max_batch_tokens = (
            max_batch_tokens  # Max total number of tokens allowed in a batch
        )

    def iter_batches(self) -> Iterator[list[tuple[int, dict[str, Any]]]]:
        """
        Yield batches of (original_index, example) pairs under the configured limits.
        Each batch is a list of (dataset index, example) tuples.
        """
        batch: list[tuple[int, dict[str, Any]]] = (
            []
        )  # Holds examples for the current batch
        token_budget = 0  # Tracks total token count for the current batch

        # Iterate through all dataset indices in the specified order
        for original_idx in self.sorted_indices:
            example = self.dataset[original_idx]  # Retrieve the dataset example
            prompt_length = len(
                example["input_ids"]
            )  # Count how many tokens this example uses

            # --- Helper function to check if adding another example would exceed limits ---
            def exceeds_limits(current_size: int, current_tokens: int) -> bool:
                # If we've reached the max number of examples per batch, stop.
                if current_size >= self.max_batch_size:
                    return True
                # If a max token limit is defined and adding this example would go over, stop.
                if (
                    self.max_batch_tokens > 0
                    and current_tokens + prompt_length > self.max_batch_tokens
                ):
                    return True
                # Otherwise, it's safe to add this example
                return False

            # --- If current batch already has data and adding this example would exceed limits ---
            if batch and exceeds_limits(len(batch), token_budget):
                yield batch  # Emit the current batch
                batch = []  # Start a new batch
                token_budget = 0  # Reset token counter

            # --- Handle the case where one example alone is too large for the token limit ---
            if (
                not batch  # Only do this if no batch is currently being built
                and self.max_batch_tokens > 0
                and prompt_length > self.max_batch_tokens
            ):
                # Emit this single example as its own batch (can't be combined with others)
                yield [(original_idx, example)]
                continue  # Move to the next example

            # --- Otherwise, add this example to the current batch ---
            batch.append((original_idx, example))
            token_budget += prompt_length  # Update running total of tokens

        # --- After processing all examples, emit any remaining items as the final batch ---
        if batch:
            yield batch


class HFEvaluator:
    """High-level orchestrator that loads data, batches prompts, and runs generation."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.model, self.tokenizer, self.device, self.max_model_len = (
            self._build_model_and_tokenizer()
        )
        self._static_cache_enabled = True
        self._compile_enabled = self.config.enable_compile
        self._compile_config = (
            CompileConfig(dynamic=True, backend="inductor", mode="reduce-overhead")
            if self._compile_enabled
            else None
        )
        if self._compile_enabled:
            # Avoid excessive recompiles on modules that expose integer attrs (e.g. layer_idx).
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
        self.collator = LeftPaddingCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

    def run(self) -> None:
        """Execute the evaluation pipeline from dataset loading to JSONL output."""
        dataset = self._load_dataset()
        sorted_indices = self._sorted_indices(dataset)
        prompts = [dataset[idx]["prompt"] for idx in range(len(dataset))]
        labels = [dataset[idx]["label"] for idx in range(len(dataset))]
        input_ids_list = [dataset[idx]["input_ids"] for idx in range(len(dataset))]

        tokens_by_index: dict[int, list[list[int]]] = {
            idx: [[] for _ in range(self.config.num_samples)]
            for idx in range(len(dataset))
        }
        stopped_by_index: dict[int, list[bool]] = {
            idx: [False for _ in range(self.config.num_samples)]
            for idx in range(len(dataset))
        }

        finished_records: dict[int, GenerationRecord] = {}
        pending_indices = list(range(len(dataset)))

        if self.config.resume:
            resume_state = _load_progress(
                output_path=self.config.output_path,
                num_samples=self.config.num_samples,
            )
            _apply_resume_state(
                resume_state=resume_state,
                tokens_by_index=tokens_by_index,
                stopped_by_index=stopped_by_index,
            )
            finished_indices = resume_state["finished_indices"]
            for index in finished_indices:
                finished_records[index] = _build_record(
                    index=index,
                    dataset_name=self.config.dataset_name,
                    prompt=prompts[index],
                    label=labels[index],
                    input_ids=input_ids_list[index],
                    prediction_token_ids=tokens_by_index[index],
                    tokenizer=self.tokenizer,
                    stopped=stopped_by_index[index],
                )
            pending_indices = [
                idx
                for idx in range(len(dataset))
                if idx not in finished_indices
            ]

        chunk_size = self.config.chunk_size
        current_max = min(chunk_size, self.config.max_new_tokens)

        while pending_indices and current_max <= self.config.max_new_tokens:
            pending_set = set(pending_indices)
            run_indices = [idx for idx in sorted_indices if idx in pending_set]

            progress = tqdm(
                total=len(run_indices),
                desc=f"Evaluating (max_new_tokens={current_max})",
                unit="sample",
            )

            new_pending: list[int] = []

            for batch_examples in _iter_generation_batches(
                dataset=dataset,
                indices=run_indices,
                tokens_by_index=tokens_by_index,
                stopped_by_index=stopped_by_index,
                num_samples=self.config.num_samples,
                max_new_tokens=current_max,
                max_batch_size=self.config.max_batch_size,
                max_batch_tokens=self.config.max_batch_tokens,
            ):
                batch_results = self._generate_for_batch(
                    batch_examples, max_new_tokens=current_max
                )
                for result in batch_results:
                    _update_tokens_from_result(
                        result=result,
                        tokens_by_index=tokens_by_index,
                        stopped_by_index=stopped_by_index,
                        max_new_tokens=current_max,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                progress.update(len(batch_examples))

            progress.close()

            for index in run_indices:
                finished = _is_finished(
                    tokens_by_index[index],
                    stopped_by_index[index],
                    current_max=current_max,
                    max_new_tokens=self.config.max_new_tokens,
                )
                record = _build_record(
                    index=index,
                    dataset_name=self.config.dataset_name,
                    prompt=prompts[index],
                    label=labels[index],
                    input_ids=input_ids_list[index],
                    prediction_token_ids=tokens_by_index[index],
                    tokenizer=self.tokenizer,
                    stopped=stopped_by_index[index],
                )
                if finished:
                    finished_records[index] = record
                else:
                    new_pending.append(index)

            _write_grouped_outputs(
                self.config.output_path,
                finished_records,
                tokens_by_index,
                stopped_by_index,
                prompts,
                labels,
                input_ids_list,
                pending_indices=new_pending,
                tokenizer=self.tokenizer,
                dataset_name=self.config.dataset_name,
            )

            pending_indices = new_pending
            if current_max >= self.config.max_new_tokens:
                break
            current_max = min(current_max + chunk_size, self.config.max_new_tokens)

        _write_final_output(
            self.config.output_path,
            finished_records,
            tokens_by_index,
            stopped_by_index,
            prompts,
            labels,
            input_ids_list,
            pending_indices,
            tokenizer=self.tokenizer,
            dataset_name=self.config.dataset_name,
        )

    def _build_model_and_tokenizer(
        self,
    ) -> tuple[LoRAMixerFFN, PreTrainedTokenizerBase, torch.device, int]:
        """Instantiate the tokenizer, base model, and LoRA-Mixer wrapper."""
        torch_dtype = resolve_dtype(bf16=self.config.bf16, fp16=self.config.fp16)
        device = torch.device(self.config.device)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer must provide a pad_token or eos_token.")
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            dtype=torch_dtype,
        )
        base_model.config.use_cache = True
        for param in base_model.parameters():
            param.requires_grad = False

        num_layers = getattr(base_model.config, "num_hidden_layers", None)
        if num_layers is None and self.config.num_layers is None:
            raise ValueError("Base model config does not expose num_hidden_layers.")
        num_layers = self.config.num_layers or num_layers

        mixer = LoRAMixerFFN(
            base_model=base_model,
            expert_paths=[str(path) for path in self.config.expert_paths],
            num_layers=num_layers,
            top_k=self.config.router_top_k,
            alpha=self.config.router_alpha,
            token_gamma=self.config.router_token_gamma,
            sequence_gamma=self.config.router_sequence_gamma,
            freeze_router=True,
            freeze_experts=True,
            enable_lora_attn=self.config.enable_lora_attn,
            normalize_router_weights=self.config.normalize_router_weights,
            lora_kwargs=self.config.lora_kwargs,
            jitter_noise=self.config.jitter_noise,
            apply_hard=self.config.apply_hard,
            router_shared_across_layers=self.config.router_shared_across_layers,
        )
        load_lora_mixer_weights(mixer, self.config.lora_path)
        mixer.to(device)
        mixer.eval()

        max_model_len = get_max_length(model=base_model)
        if max_model_len is None:
            max_model_len = getattr(tokenizer, "model_max_length", None)

        if max_model_len is None:
            raise ValueError("Unable to determine model maximum sequence length.")

        if self.config.max_length is not None:
            max_model_len = min(max_model_len, self.config.max_length)

        return mixer, tokenizer, device, max_model_len

    def _load_dataset(self):
        """Materialize the evaluation dataset via the shared tokenizer."""
        return get_dataset(
            dataset_name=self.config.dataset_name,
            dataset_path=str(self.config.dataset_path),
            tokenizer=self.tokenizer,
            max_length=self.max_model_len,
        )

    def _sorted_indices(self, dataset) -> list[int]:
        """Return dataset indices ordered by prompt length for efficient batching."""
        return sorted(
            range(len(dataset)), key=lambda idx: len(dataset[idx]["input_ids"])
        )

    def _generate_for_batch(
        self,
        batch_examples: list[tuple[int, int, dict[str, Any], int]],
        *,
        max_new_tokens: int,
    ) -> list[dict[str, Any]]:
        """Generate continuations for a batch of prompt+prefix inputs."""
        if not batch_examples:
            return []

        indices = [idx for idx, _, _, _ in batch_examples]
        sample_indices = [sample_idx for _, sample_idx, _, _ in batch_examples]
        remaining_tokens = [remaining for _, _, _, remaining in batch_examples]
        examples = [example for _, _, example, _ in batch_examples]
        batch = self.collator(examples)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_lengths = batch["prompt_lengths"]

        do_sample = (
            self.config.temperature > 0.0
            or self.config.top_p is not None
            or self.config.top_k is not None
            or self.config.num_samples > 1
        )

        batch_max_new = min(max(remaining_tokens), max_new_tokens)
        generation_kwargs = {
            "max_new_tokens": batch_max_new,
            "temperature": self.config.temperature,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": do_sample,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
        }
        if self.config.top_p is not None:
            generation_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            generation_kwargs["top_k"] = self.config.top_k

        use_static_cache = self._static_cache_enabled
        use_compile = self._compile_enabled

        with torch.no_grad():
            while True:
                run_kwargs = dict(generation_kwargs)
                run_kwargs["disable_compile"] = not use_compile
                if use_compile and self._compile_config is not None:
                    run_kwargs["compile_config"] = self._compile_config

                if use_static_cache:
                    max_cache_len = input_ids.shape[1] + batch_max_new
                    run_kwargs["past_key_values"] = StaticCache(
                        config=self.model.wrapped_model.config,
                        max_cache_len=max_cache_len,
                    )

                try:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **run_kwargs,
                    )
                    break
                except Exception as exc:
                    if use_static_cache:
                        use_static_cache = False
                        self._static_cache_enabled = False
                        warnings.warn(
                            f"StaticCache disabled after generation failure: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    if use_compile:
                        use_compile = False
                        self._compile_enabled = False
                        warnings.warn(
                            f"Generation compile disabled after failure: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    raise

        sequences = outputs.sequences.cpu()
        padded_input_length = batch["input_ids"].shape[1]
        results: list[dict[str, Any]] = []

        for row_idx in range(sequences.size(0)):
            prompt_length = prompt_lengths[row_idx].item()
            pad_length = padded_input_length - prompt_length
            seq = sequences[row_idx].tolist()
            offset = pad_length + prompt_length
            new_tokens = seq[offset:]

            results.append(
                {
                    "index": indices[row_idx],
                    "sample_idx": sample_indices[row_idx],
                    "remaining": remaining_tokens[row_idx],
                    "new_tokens": new_tokens,
                }
            )

        return results

    def _detokenize_predictions(
        self, prediction_token_ids: Sequence[Sequence[int]]
    ) -> list[str]:
        """Convert token ID predictions to strings, stopping at the first EOS token."""
        eos_id = self.tokenizer.eos_token_id
        predictions: list[str] = []

        for token_ids in prediction_token_ids:
            trimmed_tokens = token_ids
            if eos_id is not None:
                for idx, token_id in enumerate(token_ids):
                    if token_id == eos_id:
                        trimmed_tokens = token_ids[:idx]
                        break

            text = self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
            predictions.append(text)

        return predictions

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the HF evaluator script."""
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA-Mixer adapters via HF generate with dynamic batching.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the LoRA-Mixer training config JSON.",
    )
    parser.add_argument(
        "--dataset-name", required=True, help="Logical dataset identifier."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to the evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL file for generations.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completions to produce per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature forwarded to HF generate.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty applied during generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device identifier.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load the base model in float16 precision.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Load the base model in bfloat16 precision.",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=None,
        help="Pad sequences to a multiple for tensor-core efficiency.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum number of prompts per dynamic batch.",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=4096,
        help="Maximum token budget per dynamic batch (0 disables token limit).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Increment size for staged generation.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resuming from existing finished/unfinished outputs.",
    )
    parser.add_argument(
        "--allow-checkpoint",
        action="store_false",
        dest="require_final_model",
        help="Allow fallback to latest checkpoint when final_model is missing.",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Enable transformers auto-compile during generation (disabled by default).",
    )
    parser.set_defaults(resume=True)
    parser.set_defaults(require_final_model=True)
    return parser.parse_args()


def _get_finish_reason(stopped: bool) -> str:
    """Map a stop flag to a finish_reason string."""
    return "stop" if stopped else "length"


def _load_progress(
    *,
    output_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    """Load resume state from finished/unfinished JSONL outputs."""
    finished_path = _finished_output_path(output_path)
    unfinished_path = _unfinished_output_path(output_path)

    finished = _load_records_with_index(finished_path, num_samples=num_samples)
    unfinished = _load_records_with_index(unfinished_path, num_samples=num_samples)

    finished_indices = sorted(finished.keys())
    unfinished_indices = sorted(unfinished.keys())

    return {
        "finished": finished,
        "unfinished": unfinished,
        "finished_indices": finished_indices,
        "unfinished_indices": unfinished_indices,
    }


def _load_records_with_index(
    path: Path, *, num_samples: int
) -> dict[int, dict[str, Any]]:
    """Load JSONL records keyed by index and normalized for num_samples."""
    if not path.exists():
        return {}

    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "index" not in payload:
                raise ValueError(f"Missing index in {path}")
            index = int(payload["index"])
            pred_ids = payload.get("prediction_token_ids") or []
            finish_reason = payload.get("finish_reason") or []
            pred_ids, finish_reason = _normalize_prediction_state(
                pred_ids, finish_reason, num_samples
            )
            records[index] = {
                "prediction_token_ids": pred_ids,
                "finish_reason": finish_reason,
            }
    return records


def _normalize_prediction_state(
    prediction_token_ids: Sequence[Sequence[int]],
    finish_reason: Sequence[str],
    num_samples: int,
) -> tuple[list[list[int]], list[str]]:
    """Pad/trim prediction arrays to match num_samples."""
    pred_list = [list(tokens) for tokens in prediction_token_ids]
    reason_list = list(finish_reason)

    if len(pred_list) < num_samples:
        pred_list.extend([[] for _ in range(num_samples - len(pred_list))])
    if len(pred_list) > num_samples:
        pred_list = pred_list[:num_samples]

    if len(reason_list) < num_samples:
        reason_list.extend(["length" for _ in range(num_samples - len(reason_list))])
    if len(reason_list) > num_samples:
        reason_list = reason_list[:num_samples]

    return pred_list, reason_list


def _apply_resume_state(
    *,
    resume_state: dict[str, Any],
    tokens_by_index: dict[int, list[list[int]]],
    stopped_by_index: dict[int, list[bool]],
) -> None:
    """Populate in-memory state from a loaded resume snapshot."""
    for group in ("finished", "unfinished"):
        records = resume_state.get(group, {})
        for index, payload in records.items():
            tokens_by_index[index] = payload["prediction_token_ids"]
            stopped_by_index[index] = [
                reason == "stop" for reason in payload["finish_reason"]
            ]


def _finished_output_path(output_path: Path) -> Path:
    """Return the derived *.finished.jsonl path for an output file."""
    if output_path.suffixes:
        suffix = "".join(output_path.suffixes)
        stem = output_path.name[: -len(suffix)]
    else:
        suffix = ".jsonl"
        stem = output_path.name
    return output_path.with_name(f"{stem}.finished{suffix}")


def _unfinished_output_path(output_path: Path) -> Path:
    """Return the derived *.unfinished.jsonl path for an output file."""
    if output_path.suffixes:
        suffix = "".join(output_path.suffixes)
        stem = output_path.name[: -len(suffix)]
    else:
        suffix = ".jsonl"
        stem = output_path.name
    return output_path.with_name(f"{stem}.unfinished{suffix}")


def _output_already_processed(output_path: Path, *, resume: bool) -> bool:
    """Return True if the final JSONL output already exists and cannot be resumed."""
    if not output_path.exists():
        return False
    unfinished_path = _unfinished_output_path(output_path)
    if resume and unfinished_path.exists() and unfinished_path.stat().st_size > 0:
        print(
            "Output file already exists, but unfinished outputs found at "
            f"{unfinished_path}. Resuming."
        )
        return False
    print(f"Output file already exists: {output_path}. Skipping.")
    return True


def _write_grouped_outputs(
    output_path: Path,
    finished_records: dict[int, GenerationRecord],
    tokens_by_index: dict[int, list[list[int]]],
    stopped_by_index: dict[int, list[bool]],
    prompts: Sequence[str],
    labels: Sequence[Any],
    input_ids_list: Sequence[list[int]],
    *,
    pending_indices: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
) -> None:
    """Write finished and unfinished JSONL outputs for the current stage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    finished_path = _finished_output_path(output_path)
    unfinished_path = _unfinished_output_path(output_path)

    with finished_path.open("w", encoding="utf-8") as handle:
        for index in sorted(finished_records):
            handle.write(finished_records[index].to_json() + "\n")

    with unfinished_path.open("w", encoding="utf-8") as handle:
        for index in sorted(pending_indices):
            record = _build_record(
                index=index,
                dataset_name=dataset_name,
                prompt=prompts[index],
                label=labels[index],
                input_ids=input_ids_list[index],
                prediction_token_ids=tokens_by_index[index],
                tokenizer=tokenizer,
                stopped=stopped_by_index[index],
            )
            handle.write(record.to_json() + "\n")


def _write_final_output(
    output_path: Path,
    finished_records: dict[int, GenerationRecord],
    tokens_by_index: dict[int, list[list[int]]],
    stopped_by_index: dict[int, list[bool]],
    prompts: Sequence[str],
    labels: Sequence[Any],
    input_ids_list: Sequence[list[int]],
    pending_indices: Sequence[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
) -> None:
    """Write the final combined JSONL output in original dataset order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        all_indices = sorted(set(finished_records) | set(pending_indices))
        for index in all_indices:
            record = finished_records.get(index)
            if record is None:
                record = _build_record(
                    index=index,
                    dataset_name=dataset_name,
                    prompt=prompts[index],
                    label=labels[index],
                    input_ids=input_ids_list[index],
                    prediction_token_ids=tokens_by_index[index],
                    tokenizer=tokenizer,
                    stopped=stopped_by_index[index],
                )
            handle.write(record.to_json() + "\n")


def _trim_to_eos(tokens: list[int], eos_id: int | None) -> tuple[list[int], bool]:
    """Trim tokens at EOS and return whether EOS was encountered."""
    if eos_id is None:
        return tokens, False
    for idx, token_id in enumerate(tokens):
        if token_id == eos_id:
            return tokens[:idx], True
    return tokens, False


def _update_tokens_from_result(
    *,
    result: dict[str, Any],
    tokens_by_index: dict[int, list[list[int]]],
    stopped_by_index: dict[int, list[bool]],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> None:
    """Append generated tokens into the per-sample state and update stop flags."""
    index = result["index"]
    sample_idx = result["sample_idx"]
    remaining = result["remaining"]
    new_tokens = result["new_tokens"][:remaining]

    trimmed, stopped = _trim_to_eos(new_tokens, eos_token_id)
    tokens_by_index[index][sample_idx].extend(trimmed)
    if stopped:
        stopped_by_index[index][sample_idx] = True

    if len(tokens_by_index[index][sample_idx]) >= max_new_tokens:
        tokens_by_index[index][sample_idx] = tokens_by_index[index][sample_idx][
            :max_new_tokens
        ]


def _is_finished(
    prediction_token_ids: list[list[int]],
    stopped_flags: list[bool],
    *,
    current_max: int,
    max_new_tokens: int,
) -> bool:
    """Determine whether a sample is fully finished for the run."""
    return all(stopped_flags)


def _build_record(
    *,
    index: int,
    dataset_name: str,
    prompt: str,
    label: Any,
    input_ids: list[int],
    prediction_token_ids: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    stopped: list[bool],
) -> GenerationRecord:
    """Assemble a GenerationRecord from token state and metadata."""
    predictions = [
        tokenizer.decode(tokens, skip_special_tokens=True)
        for tokens in prediction_token_ids
    ]
    finish_reason = [_get_finish_reason(flag) for flag in stopped]
    return GenerationRecord(
        dataset=dataset_name,
        prompt=prompt,
        input_ids=input_ids,
        label=label,
        prediction_token_ids=prediction_token_ids,
        predictions=predictions,
        finish_reason=finish_reason,
        index=index,
    )


def _iter_generation_batches(
    *,
    dataset,
    indices: Sequence[int],
    tokens_by_index: dict[int, list[list[int]]],
    stopped_by_index: dict[int, list[bool]],
    num_samples: int,
    max_new_tokens: int,
    max_batch_size: int,
    max_batch_tokens: int,
) -> Iterator[list[tuple[int, int, dict[str, Any], int]]]:
    """Yield batches of (index, sample_idx, example, remaining_tokens) items."""
    items: list[tuple[int, int, dict[str, Any], int]] = []
    token_budget = 0

    for index in indices:
        example = dataset[index]
        for sample_idx in range(num_samples):
            if stopped_by_index[index][sample_idx]:
                continue
            current_tokens = tokens_by_index[index][sample_idx]
            remaining = max_new_tokens - len(current_tokens)
            if remaining <= 0:
                continue

            input_ids = example["input_ids"] + current_tokens
            item = (
                index,
                sample_idx,
                {
                    "input_ids": input_ids,
                    "prompt": example["prompt"],
                    "label": example["label"],
                    "raw_input_ids": example["input_ids"],
                },
                remaining,
            )

            prompt_length = len(input_ids)

            def exceeds_limits(current_size: int, current_tokens: int) -> bool:
                if current_size >= max_batch_size:
                    return True
                if max_batch_tokens > 0 and current_tokens + prompt_length > max_batch_tokens:
                    return True
                return False

            if items and exceeds_limits(len(items), token_budget):
                yield items
                items = []
                token_budget = 0

            items.append(item)
            token_budget += prompt_length

    if items:
        yield items


def main() -> None:
    """CLI entry point for dynamic-batch HF evaluation."""
    install_exit_handlers()
    args = _parse_args()

    output_path = args.output.expanduser()
    if _output_already_processed(output_path, resume=args.resume):
        return

    base_cfg = MixerTrainingConfig.from_json(args.config)

    output_root = Path(base_cfg.output_dir).expanduser()
    final_model = output_root / "final_model"
    if final_model.is_dir():
        lora_path = final_model
    elif args.require_final_model:
        raise ValueError(f"final_model not found: {final_model}")
    else:
        checkpoint_root = output_root / "checkpoints"
        latest_checkpoint = load_checkpoint_path(checkpoint_root)
        if latest_checkpoint is None:
            raise ValueError("Unable to locate LoRA-Mixer checkpoint in output_dir.")
        lora_path = Path(latest_checkpoint)

    expert_paths = [Path(path).expanduser() for path in base_cfg.expert_paths]
    if not expert_paths:
        raise ValueError("Expert paths must be provided in config.")

    if base_cfg.top_k is None:
        raise ValueError("router_top_k must be provided in config.")

    enable_lora_attn = base_cfg.enable_lora_attn
    lora_kwargs = asdict(base_cfg.lora_config)
    if not enable_lora_attn:
        lora_kwargs = None

    finished_exists = _finished_output_path(output_path).exists()
    unfinished_exists = _unfinished_output_path(output_path).exists()
    resume = args.resume or finished_exists or unfinished_exists

    config = EvalConfig(
        model_name_or_path=base_cfg.model_name_or_path,
        lora_path=lora_path,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path.expanduser(),
        output_path=output_path,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        router_top_k=base_cfg.top_k,
        normalize_router_weights=base_cfg.normalize_router_weights,
        router_alpha=0.0,
        router_token_gamma=0.,
        router_sequence_gamma=0.,
        jitter_noise=base_cfg.jitter_noise,
        apply_hard=base_cfg.apply_hard,
        router_shared_across_layers=base_cfg.router_shared_across_layers,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        fp16=args.fp16 if args.fp16 else base_cfg.fp16,
        bf16=args.bf16 if args.bf16 else base_cfg.bf16,
        pad_to_multiple_of=args.pad_to_multiple_of,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
        chunk_size=args.chunk_size,
        expert_paths=expert_paths,
        enable_lora_attn=enable_lora_attn,
        lora_kwargs=lora_kwargs,
        num_layers=base_cfg.num_layers,
        max_length=base_cfg.max_length,
        resume=resume,
        enable_compile=args.enable_compile,
    )

    evaluator = HFEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
