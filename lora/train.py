"""
LoRA/train.py

This script fine-tunes a base LLM with LoRA/QLoRA adapters using the standard
`transformers.Trainer` API. The dataset is pre-tokenized to fixed-length
sequences so the default Hugging Face data collator is sufficient.
"""

import torch
import torch.nn.functional as F
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from .config import TrainingConfig
from .dataset import get_dataset, DataCollatorForSupervisedFinetuning
from .lora_config import get_lora_config
from .utils import get_max_length


def unfreeze_lora_parameters(model):
    """Manually unfreeze LoRA parameters that may have been incorrectly frozen."""
    print("Manually unfreezing LoRA parameters...")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        # Check if it's a LoRA parameter (contains 'lora' in the name)
        name = name.lower()
        if "lora" in name:
            param.requires_grad = True
            unfrozen_count += 1

    print(f"Unfroze {unfrozen_count} LoRA parameters")
    return unfrozen_count


def train(config_path):
    """
    The main training function that orchestrates the entire fine-tuning process.

    Args:
        config_path (str): Path to the JSON configuration file.
    """
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    cfg = TrainingConfig.from_json(config_path)

    if cfg.use_lora and cfg.lora_config is None:
        raise ValueError("`use_lora` is True but no `lora_config` was provided.")

    # --- 0. Set the device if QLoRA is active ---
    if cfg.use_qlora and (os.environ.get("LOCAL_RANK", None) is not None):
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            print(f"Set CUDA device to: {local_rank}")

    # --- 2. Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer EOS token set to: {tokenizer.eos_token}")

    # --- 3. Configure Model Loading (Quantization vs. Full Precision) ---
    print("Configuring model loading...")
    # model_kwargs = {"attn_implementation": "eager"}
    model_kwargs = {
        "use_cache": False,  # "device_map": "auto"
    }
    if cfg.use_qlora:
        # Configure 4-bit quantization for QLoRA
        print("Using QLoRA (4-bit quantization).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
            # llm_int8_enable_fp32_cpu_offload=True,  # Add this for large models
        )

        model_kwargs["quantization_config"] = bnb_config
    else:
        # Load in standard bfloat16 or float16 precision
        precision = "bf16" if cfg.bf16 else ("fp16" if cfg.fp16 else "fp32")
        print(f"Using standard precision: {precision}.")
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if cfg.bf16
            else (torch.float16 if cfg.fp16 else torch.float32)
        )

    # --- 4. Load Base Model ---
    print(f"Loading base model: {cfg.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)

    for param in model.parameters():
        param.requires_grad = False

    # --- 5. Configure and Apply LoRA / QLoRA ---
    lora_cfg = None
    if cfg.use_lora:
        print("Applying LoRA/QLoRA modifications...")
        # If using QLoRA, prepare the quantized model for training.
        if cfg.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
        # For standard LoRA, manually enable gradient checkpointing if requested.
        elif cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        lora_cfg = get_lora_config(cfg.lora_config)
        model = get_peft_model(model, lora_cfg)
        print("Trainable parameters after applying LoRA:")
        model.print_trainable_parameters()

    max_seq_length = get_max_length(model)

    # --- 6. Load and Prepare Dataset ---
    print(f"Loading training dataset '{cfg.dataset_name}'...")
    train_dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        dataset_path=cfg.dataset_path,
        tokenizer=tokenizer,
        max_training_sample=cfg.max_training_sample,
        seed=cfg.seed,
    )
    print(f"Training dataset size: {len(train_dataset)} samples.")

    eval_dataset = None
    if cfg.validation_dataset_path:
        eval_name = cfg.dataset_name
        print(f"Loading validation dataset '{eval_name}'...")
        eval_dataset = get_dataset(
            dataset_name=eval_name,
            dataset_path=cfg.validation_dataset_path,
            tokenizer=tokenizer,
        )
        print(f"Validation dataset size: {len(eval_dataset)} samples.")

    # --- 7. Data Collator ---
    data_collator = DataCollatorForSupervisedFinetuning(
        pad_token_id=tokenizer.pad_token_id, max_length=max_seq_length
    )

    # --- 8. Set Up Training Arguments ---
    print("Setting up training arguments...")

    eval_strategy = cfg.eval_strategy
    if cfg.validation_dataset_path and cfg.eval_strategy == "no":
        eval_strategy = "steps"

    # Prepare TrainingArguments kwargs
    training_args_kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps,
        "fp16": cfg.fp16,
        "bf16": cfg.bf16,
        "report_to": cfg.report_to,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "ddp_find_unused_parameters": False,
        "eval_strategy": eval_strategy if cfg.validation_dataset_path else "no",
        "remove_unused_columns": False,
        "warmup_ratio": cfg.warmup_ratio,
        "dataloader_num_workers": 4,  # Parallel data loading
        "optim": "paged_adamw_32bit" if cfg.use_qlora else "adamw_torch_fused",
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_strategy": "steps",
        "greater_is_better": False,
    }
    if cfg.deepspeed_config_path:
        training_args_kwargs["deepspeed"] = cfg.deepspeed_config_path
    if eval_strategy == "steps":
        training_args_kwargs["load_best_model_at_end"] = cfg.load_best_model_at_end

    if cfg.validation_dataset_path:
        training_args_kwargs["metric_for_best_model"] = "eval_loss"

    if cfg.validation_dataset_path and cfg.eval_steps is not None:
        training_args_kwargs["eval_steps"] = cfg.eval_steps
    if cfg.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = cfg.save_total_limit
    training_args = TrainingArguments(**training_args_kwargs)

    # --- 9. Initialize Trainer ---
    print("Initializing Trainer...")
    callbacks = []
    if (
        eval_dataset is not None
        and cfg.early_stopping_patience
        and cfg.early_stopping_patience > 0
    ):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # --- 9.5 Ensure that the model is trainable ---
    # if cfg.use_lora:
    #    unfreeze_lora_parameters(model)

    # Print trainable parameters for verification
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # --- 10. Verify training setup ---
    print("\nVerifying training setup...")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")
    print(
        f"Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}"
    )

    # --- 11. Start Training ---
    print("\nStarting model training with Trainer...")
    print("=" * 60)

    # Check if we should resume from checkpoint
    checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [
            os.path.join(cfg.output_dir, d)
            for d in os.listdir(cfg.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)  # latest checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    # --- 12. Save Final Model ---
    # PeftModel.save_pretrained persists only the adapter weights, which is what we want.
    if trainer.is_world_process_zero():
        final_output_dir = os.path.join(cfg.output_dir, "final_checkpoint")
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
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using transformers.Trainer with LoRA adapters."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        train(args.config)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
