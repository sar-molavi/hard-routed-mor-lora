"""Entry point for offline RL using Accelerate's batch dispatch path.

This variant:
  - Uses `accelerator_config={"dispatch_batches": True}` so the DataLoader
    lives on rank 0.
  - Builds the prefetch dataset only on rank 0 via `dispatch_prefetch_dataset`.
  - Scales DataLoader batch size by world size so the per-rank batch matches
    the standard (non-dispatch) configuration after per-rank slicing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from dataclasses import asdict
from functools import partial

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

from .config import OfflineTrainingConfig
from .dispatch_prefetch_dataset import build_distributed_prefetch_dataset
from .prefetch_rlvf_fastapi_dataset import (
    AsyncFastAPIPrefetchPipeline,
    DATASET_STATE_NAME,
)
from .reward import compute_reward
from .rlhf_collator import DataCollatorForRLFinetuning
from .trainer_dispatch import (
    DispatchSequenceLevelOfflineTrainer,
    DispatchTokenLevelOfflineTrainer,
)
from .utils import load_checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline RL LoRA fine-tuning (dispatch_batches=True)."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the offline RL configuration JSON file.",
    )
    return parser.parse_args()


def _enable_gradient_checkpointing(model) -> None:
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()


def _load_model(cfg: OfflineTrainingConfig):
    model_kwargs: dict[str, Any] = {"use_cache": False}
    if cfg.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        dtype = (
            torch.bfloat16
            if cfg.bf16
            else (torch.float16 if cfg.fp16 else torch.float32)
        )
        model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if cfg.use_lora:
        if cfg.use_qlora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
            )
            if cfg.gradient_checkpointing:
                _enable_gradient_checkpointing(model)
        elif cfg.gradient_checkpointing:
            _enable_gradient_checkpointing(model)

        lora_kwargs = asdict(cfg.lora_config)
        peft_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif cfg.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    if cfg.gradient_checkpointing and hasattr(model, "_set_static_graph"):
        try:
            model._set_static_graph()
        except Exception:
            pass

    return model


def _load_tokenizer(cfg: OfflineTrainingConfig):
    name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define a pad token or EOS token.")
    tokenizer.padding_side = "left"
    return tokenizer


def _build_prefetch_dataset(cfg: OfflineTrainingConfig):
    kwargs = cfg.prefetch.to_prefetch_kwargs()
    pipeline = AsyncFastAPIPrefetchPipeline(**kwargs)
    return build_distributed_prefetch_dataset(
        pipeline,
        output_dir=cfg.output_dir,
    )


def _select_trainer(cfg: OfflineTrainingConfig):
    loss_type = cfg.algorithm.loss_type.lower()
    if loss_type == "sequence":
        return DispatchSequenceLevelOfflineTrainer
    if loss_type in {"token", "token_dapo"}:
        return DispatchTokenLevelOfflineTrainer
    raise ValueError(f"Unsupported loss_type '{cfg.algorithm.loss_type}'.")


def _load_dataset_state(*, cfg: OfflineTrainingConfig, dataset) -> None:
    checkpoint_dir = load_checkpoint_path(cfg.output_dir)
    if checkpoint_dir is None:
        return

    path = Path(checkpoint_dir) / DATASET_STATE_NAME
    if not path.is_file():
        return

    state = torch.load(path, map_location="cpu")
    load_fn = getattr(dataset, "load_state_dict", None)
    if callable(load_fn):
        load_fn(state)


def train(config_path: str) -> None:
    cfg = OfflineTrainingConfig.from_json(config_path)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final_model"
    if final_dir.is_dir():
        print(f"Final model already exists at {final_dir}; skipping training.")
        return
    resume_checkpoint = load_checkpoint_path(output_dir)

    tokenizer = _load_tokenizer(cfg)
    model = _load_model(cfg)

    dataset = _build_prefetch_dataset(cfg)
    # _load_dataset_state(cfg=cfg, dataset=dataset)
    accelerator_config: dict[str, Any] = {"dispatch_batches": True}

    reward_fn = partial(
        compute_reward,
        correct_reward=cfg.algorithm.correct_reward,
        incorrect_reward=cfg.algorithm.incorrect_reward,
        bonus_think=cfg.algorithm.bonus_think,
        bonus_json_on_wrong=cfg.algorithm.bonus_json_on_wrong,
    )

    data_collator = DataCollatorForRLFinetuning(
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        max_length=cfg.max_sequence_length,
        reward_scaling=getattr(cfg.algorithm, "reward_scaling", None),
        advantage_mode=getattr(cfg.algorithm, "advantage_mode", "grpo"),
    )

    args_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        report_to=cfg.report_to,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        ignore_data_skip=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        accelerator_config=accelerator_config,
        # log_level="info",
    )
    args = TrainingArguments(**args_kwargs)

    trainer_cls = _select_trainer(cfg)
    trainer_kwargs = {}
    if cfg.algorithm.loss_type.lower() == "token_dapo":
        trainer_kwargs["global_token_normalization"] = True

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        ratio_clip=cfg.algorithm.ratio_clip,
        ratio_clip_high=cfg.algorithm.ratio_clip_high,
        normalize_advantages=cfg.algorithm.normalize_advantages,
        **trainer_kwargs,
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os._exit(0)
    #if trainer.is_world_process_zero():
    #    final_dir.mkdir(parents=True, exist_ok=True)
    #    trainer.save_model(str(final_dir))
    #    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
