from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

import evaluate

from .config import FineTuningConfig
from .datasets import LABEL2ID, ID2LABEL, get_dataset, DatasetCollator


def compute_metrics():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def fn(p: EvalPrediction) -> dict[str, float]:
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        metrics = {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
        }
        per_label_f1 = f1.compute(predictions=preds, references=labels, average=None)[
            "f1"
        ]
        for label_id, label_name in ID2LABEL.items():
            mask = labels == label_id
            if np.any(mask):
                label_acc = float(np.mean(preds[mask] == labels[mask]))
            else:
                label_acc = 0.0
            metrics[f"accuracy_{label_name}"] = label_acc
            metrics[f"f1_{label_name}"] = float(per_label_f1[label_id])

        return metrics

    return fn


def _get_model_and_tokenizer(
    config: FineTuningConfig,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16 if config.bf16 else None,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def train(config: FineTuningConfig):
    set_seed(config.seed)

    train_set = get_dataset(
        info=config.dataset_info,
        max_samples=config.max_train_samples,
        seed=config.seed,
    )

    val_set = None
    if config.validation_dataset_info:
        val_set = get_dataset(
            info=config.validation_dataset_info,
        )

    model, tokenizer = _get_model_and_tokenizer(config)

    collator = DatasetCollator(
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    eval_strategy = config.eval_strategy
    if val_set and eval_strategy == "no":
        eval_strategy = "steps"

    training_args_kwargs = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "bf16": config.bf16,
        "report_to": config.report_to,
        "gradient_checkpointing": config.gradient_checkpointing,
        "ddp_find_unused_parameters": False,
        "eval_strategy": eval_strategy,
        "remove_unused_columns": False,
        "warmup_ratio": config.warmup_ratio,
        "dataloader_num_workers": 4,  # Parallel data loading
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "logging_strategy": "steps",
        "greater_is_better": True,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": "f1_macro",
    }

    if config.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = config.save_total_limit

    training_args = TrainingArguments(
        **training_args_kwargs,
    )

    # --- 9. Initialize Trainer ---
    print("Initializing Trainer...")
    callbacks = []
    if (
        val_set is not None
        and config.early_stopping_patience
        and config.early_stopping_patience > 0
    ):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=compute_metrics(),
    )

    # --- 10. Verify training setup ---
    print("\nVerifying training setup...")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Training samples: {len(train_set)}")
    if val_set:
        print(f"Validation samples: {len(val_set)}")
    print(
        f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else config.per_device_train_batch_size * config.gradient_accumulation_steps}"
    )

    # --- 11. Start Training ---
    print("\nStarting model training with Trainer...")
    print("=" * 60)

    # Check if we should resume from checkpoint
    checkpoint = None
    if os.path.isdir(config.output_dir):
        checkpoints = [
            os.path.join(config.output_dir, d)
            for d in os.listdir(config.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)  # latest checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    # --- 12. Save Final Model ---
    # PeftModel.save_pretrained persists only the adapter weights, which is what we want.
    final_output_dir = os.path.join(config.output_dir, "final_checkpoint")
    print(f"\nSaving final LoRA adapter to {final_output_dir}...")
    trainer.save_model(final_output_dir)

    # Save tokenizer as well
    tokenizer.save_pretrained(final_output_dir)

    print("=" * 60)
    print(
        f"Training complete! Final LoRA adapter saved to: {os.path.abspath(final_output_dir)}"
    )
    print("You can now use this adapter for inference or further training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    config = FineTuningConfig.from_json(args.config)
    train(config)
