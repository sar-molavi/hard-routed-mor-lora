"""Run vLLM prediction for mixed prompts using classifier-routed LoRAs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

from lora_offline.utils import get_max_length

from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


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
    routing: dict[str, Any]

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
                "routing": self.routing,
            },
            ensure_ascii=False,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    logger.info("Reading routing manifest: %s", path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Routing manifest is empty: {path}")
    logger.info("Loaded %d routing rows", len(rows))
    return rows


def _write_jsonl(path: Path, records: list[GenerationRecord]) -> None:
    logger.info("Writing vLLM predictions: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.index):
            handle.write(record.to_json() + "\n")
    logger.info("Wrote %d prediction rows", len(records))


def _format_prompt(prompt: str, tokenizer: PreTrainedTokenizerBase) -> str:
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (tokenizer.bos_token or "") + prompt + "\n\n---\n\nAnswer:\n"


def _tokenize_rows(
    rows: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_model_len: int,
) -> list[dict[str, Any]]:
    logger.info("Formatting and tokenizing %d routed mixed prompts", len(rows))
    tokenized_rows: list[dict[str, Any]] = []
    too_long: list[tuple[int, int]] = []
    for row in tqdm(rows, desc="Tokenizing routed prompts", unit="prompt"):
        formatted_prompt = _format_prompt(str(row["prompt"]), tokenizer)
        input_ids = tokenizer(
            formatted_prompt,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(input_ids) > max_model_len:
            too_long.append((int(row["mixed_index"]), len(input_ids)))
        tokenized = dict(row)
        tokenized["formatted_prompt"] = formatted_prompt
        tokenized["input_ids"] = input_ids
        tokenized_rows.append(tokenized)

    if too_long:
        first_index, first_len = too_long[0]
        raise ValueError(
            f"{len(too_long)} prompts exceed max_model_len={max_model_len}; "
            f"first mixed_index={first_index}, length={first_len}."
        )
    return tokenized_rows


def _adapter_key(row: dict[str, Any], *, fallback_to_base: bool) -> str | None:
    adapter = row.get("predicted_adapter")
    if adapter:
        return str(adapter)

    for candidate in row.get("adapter_candidates") or []:
        candidate_adapter = candidate.get("adapter")
        if candidate_adapter:
            logger.info(
                "mixed_index=%s predicted_label=%s has no direct adapter; "
                "falling back to ranked label=%s adapter=%s",
                row.get("mixed_index"),
                row.get("predicted_label"),
                candidate.get("label"),
                candidate_adapter,
            )
            return str(candidate_adapter)

    if fallback_to_base:
        return None
    raise ValueError(
        "Routing row has no predicted_adapter and no adapter_candidates entry with "
        "an adapter path. Pass --allow-base-fallback or create the routing manifest "
        "with --adapter-map."
    )


def _build_lora_requests(adapter_paths: list[str]) -> dict[str, LoRARequest]:
    requests: dict[str, LoRARequest] = {}
    for adapter_id, adapter_path in enumerate(sorted(adapter_paths), start=1):
        path = Path(adapter_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LoRA adapter path does not exist: {path}")
        requests[adapter_path] = LoRARequest(
            lora_name=path.name or f"adapter_{adapter_id}",
            lora_int_id=adapter_id,
            lora_path=str(path),
        )
    logger.info("Prepared %d LoRA requests", len(requests))
    return requests


def _group_rows_by_adapter(
    rows: list[dict[str, Any]], *, fallback_to_base: bool
) -> dict[str | None, list[dict[str, Any]]]:
    groups: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        key = _adapter_key(row, fallback_to_base=fallback_to_base)
        groups.setdefault(key, []).append(row)
    logger.info(
        "Grouped routed prompts by adapter: %s",
        {str(key): len(value) for key, value in groups.items()},
    )
    return groups


def _make_record(
    *,
    row: dict[str, Any],
    output,
    tokenizer: PreTrainedTokenizerBase,
) -> GenerationRecord:
    token_seqs = [list(candidate.token_ids) for candidate in output.outputs]
    predictions = [
        tokenizer.decode(tokens, skip_special_tokens=True) for tokens in token_seqs
    ]
    finish_reasons = [candidate.finish_reason for candidate in output.outputs]
    selected_adapter = _adapter_key(row, fallback_to_base=True)

    return GenerationRecord(
        dataset="mixed_gsm8k_boolq",
        index=int(row["mixed_index"]),
        prompt=row["formatted_prompt"],
        input_ids=row["input_ids"],
        label=row.get("label"),
        predictions=predictions,
        prediction_token_ids=token_seqs,
        finish_reasons=finish_reasons,
        routing={
            "classification_index": row.get("classification_index"),
            "classification_prompt_order": row.get("classification_prompt_order"),
            "mixed_prompt_order": row.get("mixed_prompt_order"),
            "predicted_label": row.get("predicted_label"),
            "predicted_adapter": row.get("predicted_adapter"),
            "selected_adapter": selected_adapter,
            "ranked_labels": row.get("ranked_labels"),
            "adapter_candidates": row.get("adapter_candidates"),
            "source": row.get("source"),
        },
    )


def run_vllm_prediction(
    *,
    model_name: str,
    routing_manifest_path: Path,
    output_path: Path,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    gpu_memory_utilization: float,
    swap_space: int,
    cpu_offload_gb: int,
    no_prefix_caching: bool,
    no_eager: bool,
    max_lora_rank: int,
    allow_base_fallback: bool,
    max_model_len: int | None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token_id = tokenizer.eos_token_id
    stop_token_ids = [eos_token_id] if eos_token_id is not None else None

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    resolved_max_model_len = get_max_length(model=None, config=hf_config)
    if resolved_max_model_len is None:
        resolved_max_model_len = getattr(tokenizer, "model_max_length", None)
    if resolved_max_model_len is None:
        raise ValueError("Unable to determine model max length.")
    if max_model_len is not None:
        resolved_max_model_len = min(resolved_max_model_len, max_model_len)

    rows = _tokenize_rows(
        _read_jsonl(routing_manifest_path),
        tokenizer=tokenizer,
        max_model_len=resolved_max_model_len,
    )
    groups = _group_rows_by_adapter(rows, fallback_to_base=allow_base_fallback)
    adapter_paths = [key for key in groups if key is not None]
    lora_requests = _build_lora_requests(adapter_paths)

    logger.info("Initializing vLLM model: %s", model_name)
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        enable_lora=bool(adapter_paths),
        max_model_len=resolved_max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        cpu_offload_gb=cpu_offload_gb,
        enforce_eager=not no_eager,
        enable_prefix_caching=not no_prefix_caching,
        max_lora_rank=max_lora_rank if adapter_paths else None,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        detokenize=False,
        n=num_samples,
    )

    records: list[GenerationRecord] = []
    for adapter_path, group_rows in groups.items():
        lora_request = lora_requests.get(adapter_path) if adapter_path else None
        logger.info(
            "Generating %d prompts with adapter=%s",
            len(group_rows),
            adapter_path or "base",
        )
        prompts = [
            TokensPrompt(prompt_token_ids=row["input_ids"]) for row in group_rows
        ]
        outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
        )
        for row, output in zip(group_rows, outputs):
            records.append(_make_record(row=row, output=output, tokenizer=tokenizer))

    _write_jsonl(output_path, records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM prediction for classifier-routed mixed prompts."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--routing-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=int, default=20)
    parser.add_argument("--cpu-offload-gb", type=int, default=0)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--no-eager", action="store_true")
    parser.add_argument("--allow-base-fallback", action="store_true")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional cap for vLLM max_model_len.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_vllm_prediction(
        model_name=args.model_name,
        routing_manifest_path=args.routing_manifest.expanduser(),
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
        allow_base_fallback=args.allow_base_fallback,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
