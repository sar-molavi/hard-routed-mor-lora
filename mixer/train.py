"""Training entrypoint for the updated LoRA-Mixer pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from dataclasses import asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from .config import MixerTrainingConfig
from .dataset import get_dataset, DataCollatorForSupervisedFinetuning

from .mixer import LoRAMixerFFN
from .trainer import BalancedLoRATrainer, GumbelTemperatureCallback
from .utils import (
    clone_lora_parameters,
    print_trainable_parameters,
    resolve_dtype,
    save_lora_mixer_weights,
    unwrap_model,
    get_max_length,
    load_checkpoint_path,
    install_exit_handlers,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LoRA-Mixer model.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to mixer JSON config."
    )
    parser.add_argument(
        "-ds",
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to mixer DeepSpeed config.",
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
    return tokenizer


def _load_base_model(config: MixerTrainingConfig) -> AutoModelForCausalLM:
    model_kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(bf16=config.bf16, fp16=config.fp16),
        "use_cache": False,
        "attn_implementation": "eager",
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, **model_kwargs
    )
    base_model.config.use_cache = False
    for param in base_model.parameters():
        param.requires_grad = False
    return base_model


def _uses_gumbel_softmax(router_mode: str | None) -> bool:
    return (router_mode or "").lower() in {"gumbel", "gumbel_softmax"}


def train(
    config_path: str,
    *,
    max_length_override: int | None = None,
    deepspeed_config: str | None = None,
):
    cfg = MixerTrainingConfig.from_json(config_path)

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(cfg.model_name_or_path)
    base_model = _load_base_model(cfg)

    max_length = max_length_override or cfg.max_length or get_max_length(base_model)
    if max_length is None:
        raise ValueError("Unable to determine model max sequence length.")

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

    print_trainable_parameters(model)

    train_dataset = get_dataset(
        cfgs=cfg.train_set_configs,
        tokenizer=tokenizer,
        apply_class_weight=cfg.apply_class_weight,
        n_repeats=cfg.n_repeats,
        seed=cfg.seed,
    )
    eval_dataset = None
    if cfg.eval_set_configs:
        eval_dataset = get_dataset(
            cfgs=cfg.eval_set_configs,
            tokenizer=tokenizer,
            apply_class_weight=cfg.apply_class_weight,
            seed=cfg.seed,
        )

    data_collator = DataCollatorForSupervisedFinetuning(
        pad_token_id=tokenizer.pad_token_id
    )

    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_kwargs = dict(
        output_dir=str(output_root / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=cfg.remove_unused_columns,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        dataloader_num_workers=cfg.dataloader_num_workers,
        load_best_model_at_end=cfg.load_best_model_at_end and eval_dataset is not None,
        max_steps=cfg.max_steps,
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
    )
    cfg.deepspeed_config = cfg.deepspeed_config or deepspeed_config
    if cfg.deepspeed_config:
        deepspeed_path = Path(cfg.deepspeed_config)
        if not deepspeed_path.is_file():
            raise ValueError(f"DeepSpeed config not found at {deepspeed_path!s}")
        training_kwargs["deepspeed"] = str(deepspeed_path)
    if eval_dataset is not None and cfg.eval_steps is not None:
        training_kwargs["eval_steps"] = cfg.eval_steps
    training_args = TrainingArguments(**training_kwargs)

    old_expert_params = (
        None
        if (cfg.freeze_experts or cfg.distill_l2_reg <= 0)
        else clone_lora_parameters(model)
    )

    trainer = BalancedLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        balance_loss_weight=cfg.balance_loss_weight,
        use_default_loss=cfg.use_default_loss,
        distill_l2_reg=cfg.distill_l2_reg,
        old_expert_params=old_expert_params,
        unconstrained_experts=cfg.unconstrained_experts,
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

    checkpoint = load_checkpoint_path(training_args.output_dir)
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_world_process_zero():
        final_dir = output_root / "final_model"
        base_model = unwrap_model(model)
        save_lora_mixer_weights(base_model, final_dir)


def main():
    install_exit_handlers()
    args = _parse_args()
    train(
        args.config,
        max_length_override=args.max_length_override,
        deepspeed_config=args.deepspeed_config,
    )


if __name__ == "__main__":
    main()
