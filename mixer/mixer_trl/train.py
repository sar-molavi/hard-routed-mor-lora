"""Training entrypoint for GRPO training with LoRA-Mixer."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from functools import partial
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixer.mixer import LoRAMixerFFN
from mixer.utils import (
    get_max_length,
    install_exit_handlers,
    load_checkpoint_path,
    load_lora_mixer_weights,
    print_trainable_parameters,
    resolve_dtype,
)
from mixer.mixer_trl.config import MixerGRPOTrainingConfig
from mixer.mixer_trl.dataset import build_grpo_dataset
from mixer.mixer_trl.hf_adapter import HFCompatibleMixerModel
from lora_offline.reward import compute_reward as offline_compute_reward
from mixer.mixer_trl.trainer import build_grpo_trainer
from mixer.trainer import GumbelTemperatureCallback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA-Mixer with TRL GRPO.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to GRPO JSON config.")
    parser.add_argument(
        "-ds",
        "--deepspeed-config",
        type=str,
        default=None,
        help="Optional DeepSpeed config path override.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        dest="max_length_override",
        help="Optional override for tokenizer max sequence length.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DeepSpeed/torchrun injected argument; ignored by this script.",
    )
    return parser.parse_args()


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_model(config: MixerGRPOTrainingConfig) -> AutoModelForCausalLM:
    model_kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(bf16=config.bf16, fp16=config.fp16),
        "use_cache": False,
        "attn_implementation": "eager",
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    base_model.config.use_cache = False

    for param in base_model.parameters():
        param.requires_grad = False

    return base_model


def _uses_gumbel_softmax(router_mode: str | None) -> bool:
    return (router_mode or "").lower() in {"gumbel", "gumbel_softmax"}


def _enable_input_require_grads_if_needed(model, *, gradient_checkpointing: bool) -> None:
    if not gradient_checkpointing:
        return

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Fallback/guard for wrappers where `enable_input_require_grads` is unavailable
    # or ineffective in nested wrapped models.
    candidates = (
        model,
        getattr(model, "mixer_model", None),
        getattr(getattr(model, "mixer_model", None), "wrapped_model", None),
        getattr(getattr(model, "mixer_model", None), "wrapped_base_model", None),
    )
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "get_input_embeddings"):
            continue
        embeddings = candidate.get_input_embeddings()
        if embeddings is None:
            continue

        hook_name = "_mixer_require_grads_hook"
        if getattr(embeddings, hook_name, None) is None:
            handle = embeddings.register_forward_hook(
                lambda _module, _inp, out: out.requires_grad_(True)
            )
            setattr(embeddings, hook_name, handle)
        return


def _build_reward_fn(cfg: MixerGRPOTrainingConfig):
    base_reward_fn = partial(
        offline_compute_reward,
        correct_reward=cfg.reward.correct_reward,
        incorrect_reward=cfg.reward.incorrect_reward,
        bonus_think=cfg.reward.bonus_think,
        bonus_json_on_wrong=cfg.reward.bonus_json_on_wrong,
    )

    def reward_fn(
        *,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        ground_truth: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        return base_reward_fn(
            ground_truths=ground_truth,
            completions=completions,
            prompts=prompts,
            completion_ids=completion_ids,
            finish_reasons=kwargs.get("finish_reasons"),
            tokenizer=kwargs.get("tokenizer"),
        )

    return reward_fn


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _validate_runtime_config(cfg: MixerGRPOTrainingConfig) -> None:
    world_size = _distributed_world_size()
    if cfg.gradient_checkpointing and world_size > 1:
        warnings.warn(
            "Using gradient checkpointing in distributed training with "
            "non-reentrant checkpoints."
        )

    if cfg.deepspeed_config is not None:
        deepspeed_path = Path(cfg.deepspeed_config)
        if not deepspeed_path.is_file():
            raise ValueError(f"DeepSpeed config not found at {deepspeed_path!s}")


def train(
    config_path: str,
    *,
    max_length_override: int | None = None,
    deepspeed_config: str | None = None,
) -> None:
    cfg = MixerGRPOTrainingConfig.from_json(config_path)

    if deepspeed_config is not None:
        cfg.deepspeed_config = deepspeed_config
    _validate_runtime_config(cfg)

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(cfg.model_name_or_path)
    base_model = _load_base_model(cfg)

    max_length = max_length_override or cfg.max_length or get_max_length(base_model)
    if max_length is None:
        raise ValueError("Unable to determine model max sequence length.")
    if cfg.max_prompt_length is None:
        cfg.max_prompt_length = max_length

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
        router_mode=cfg.router_mode,
        gumbel_temperature=cfg.gumbel_temperature_scheduler.initial_temperature,
        gumbel_hard=cfg.gumbel_hard,
        router_shared_across_layers=cfg.router_shared_across_layers,
    )
    model = HFCompatibleMixerModel(model)

    if cfg.init_mixer_checkpoint:
        load_lora_mixer_weights(model.mixer_model, cfg.init_mixer_checkpoint, strict=False)

    _enable_input_require_grads_if_needed(
        model, gradient_checkpointing=cfg.gradient_checkpointing
    )

    print_trainable_parameters(model)

    train_dataset = build_grpo_dataset(
        cfgs=cfg.train_set_configs,
        tokenizer=tokenizer,
        seed=cfg.seed,
        max_samples_per_dataset=cfg.max_samples_per_dataset,
    )

    eval_dataset = None
    if cfg.eval_set_configs:
        eval_dataset = build_grpo_dataset(
            cfgs=cfg.eval_set_configs,
            tokenizer=tokenizer,
            seed=cfg.seed,
            max_samples_per_dataset=cfg.max_samples_per_dataset,
        )
    reward_fn = _build_reward_fn(cfg)

    trainer = build_grpo_trainer(
        cfg=cfg,
        output_dir=checkpoint_dir,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_fn=reward_fn,
    )
    if _uses_gumbel_softmax(cfg.router_mode):
        scheduler = cfg.gumbel_temperature_scheduler
        trainer.add_callback(
            GumbelTemperatureCallback(
                scheduler_name=scheduler.name,
                initial_temperature=scheduler.initial_temperature,
                final_temperature=scheduler.final_temperature,
                hold_steps=scheduler.hold_steps,
            )
        )

    checkpoint = load_checkpoint_path(checkpoint_dir)
    trainer.train(resume_from_checkpoint=checkpoint)
    _barrier_if_distributed()

    if trainer.is_world_process_zero():
        final_dir = output_root / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
    _barrier_if_distributed()



def main() -> None:
    install_exit_handlers()
    args = _parse_args()
    train(
        args.config,
        max_length_override=args.max_length_override,
        deepspeed_config=args.deepspeed_config,
    )


if __name__ == "__main__":
    main()
