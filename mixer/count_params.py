#!/usr/bin/env python3
"""Count total and trainable parameters from a LoRA-Mixer config JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

from transformers import AutoModelForCausalLM

from .config import MixerTrainingConfig
from .mixer import LoRAMixerFFN
from .utils import load_lora_mixer_weights, resolve_dtype


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print total/trainable parameter counts for a LoRA-Mixer config."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to MixerTrainingConfig JSON.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Optional checkpoint/final_model directory for loading lora_mixer.pth "
            "before counting."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional output path to write counts as JSON.",
    )
    return parser.parse_args()


def _load_base_model(config: MixerTrainingConfig) -> AutoModelForCausalLM:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(bf16=config.bf16, fp16=config.fp16),
        "use_cache": False,
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    return model


def _count_parameters(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def main() -> None:
    args = _parse_args()
    cfg = MixerTrainingConfig.from_json(args.config)

    base_model = _load_base_model(cfg)
    model = LoRAMixerFFN(
        base_model=base_model,
        expert_paths=cfg.expert_paths,
        num_layers=cfg.num_layers,
        alpha=cfg.router_alpha,
        token_gamma=cfg.router_token_gamma,
        sequence_gamma=cfg.router_sequence_gamma,
        freeze_router=cfg.freeze_router,
        freeze_experts=cfg.freeze_experts,
        top_k=cfg.top_k,
        enable_lora_attn=cfg.enable_lora_attn,
        lora_kwargs=asdict(cfg.lora_config),
        enable_gradient_checkpointing=cfg.gradient_checkpointing,
        normalize_router_weights=cfg.normalize_router_weights,
        jitter_noise=cfg.jitter_noise,
        apply_hard=cfg.apply_hard,
        router_shared_across_layers=cfg.router_shared_across_layers,
    )

    if args.checkpoint_dir:
        load_lora_mixer_weights(model, args.checkpoint_dir, strict=False)

    total_params, trainable_params = _count_parameters(model)
    trainable_pct = (
        (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    )

    result = {
        "config": str(Path(args.config).expanduser()),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percent": round(trainable_pct, 6),
    }

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {trainable_pct:.6f}")

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report to: {out_path}")


if __name__ == "__main__":
    main()
