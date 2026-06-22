from __future__ import annotations

import os
import random
import pickle
from typing import Any

import nevergrad as ng
import numpy as np
import torch
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file as safe_load_file
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from mixer.dataset import DataCollatorForSupervisedFinetuning, get_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """
    Set Python / NumPy / PyTorch RNG state for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype: str | None) -> torch.dtype | str:
    """
    Resolve a user-facing dtype string into a torch dtype understood by
    `from_pretrained(...)`.

    Returns:
        A torch.dtype or the string "auto".
    """
    if dtype is None or dtype == "auto":
        return "auto"

    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[dtype]


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """
    Load tokenizer and ensure padding is defined.

    For causal LMs, many tokenizers do not define a pad token. In that case,
    we fall back to EOS padding, which is standard for batched generation/training.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id even after pad_token fallback.")

    return tokenizer


def load_base_model(model_name_or_path: str, torch_dtype: str = "auto") -> nn.Module:
    """
    Load the base causal LM on the target device in eval mode.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_torch_dtype(torch_dtype),
    )
    model.to(DEVICE)
    model.eval()
    return model


def read_peft_adapter_state_dict(
    adapter_path: str, map_location: str = "cpu"
) -> dict[str, torch.Tensor]:
    """
    Read a PEFT adapter state dict from either:
      - adapter_model.safetensors
      - adapter_model.bin

    Only LoRA tensors are retained.

    Returns:
        Dict[name -> tensor] containing only entries whose name includes "lora_".
    """
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = safe_load_file(safetensors_path, device=map_location)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=map_location)
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {adapter_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )

    lora_state = {}

    for k, v in state_dict.items():
        if "lora_" not in k:
            continue

        # PEFT expects ".default" in adapter parameter names
        if ".default." not in k:
            k = k.replace("lora_A.", "lora_A.default.")
            k = k.replace("lora_B.", "lora_B.default.")

        lora_state[k] = v

    if not lora_state:
        raise ValueError(f"No LoRA tensors found in adapter: {adapter_path}")

    return lora_state


def save_checkpoint(
    checkpoint_dir: str,
    optimizer: ng.optimizers.base.Optimizer,
    step: int,
    history: list[dict[str, Any]],
) -> None:
    """
    Persist optimizer state so training can resume exactly where it stopped.

    We serialize:
        - Nevergrad optimizer object
        - current step
        - optimization history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "optimizer": optimizer,
        "step": step,
        "history": history,
    }

    path = os.path.join(checkpoint_dir, "optimizer_state.pkl")

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(checkpoint_dir: str):
    """
    Load a previously saved optimizer checkpoint.
    """
    path = os.path.join(checkpoint_dir, "optimizer_state.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    return payload["optimizer"], payload["step"], payload["history"]


def load_lora_state_dicts(
    expert_paths: list[str],
    device: str = DEVICE,
) -> tuple[list[dict[str, torch.Tensor]], str]:
    """
    Load all expert LoRA state dicts and move them to the target device once.

    This avoids repeated CPU->GPU copies during every optimization step.

    Returns:
        (list_of_lora_state_dicts, inferred_base_model_name)
    """
    if not expert_paths:
        raise ValueError("expert_paths is empty.")

    base_model_name = PeftConfig.from_pretrained(
        expert_paths[0]
    ).base_model_name_or_path

    state_dicts: list[dict[str, torch.Tensor]] = []

    for path in expert_paths:
        cfg = PeftConfig.from_pretrained(path)
        if cfg.base_model_name_or_path != base_model_name:
            raise ValueError(
                "All experts must share the same base model. "
                f"Expected {base_model_name}, got {cfg.base_model_name_or_path} from {path}."
            )

        state = read_peft_adapter_state_dict(path, map_location="cpu")

        # Keep all expert tensors on the target device from the start.
        # This is much faster than re-copying merged tensors every step.
        state = {k: v.cpu() for k, v in state.items()}
        state_dicts.append(state)

    return state_dicts, base_model_name


def validate_lora_state_dicts(
    lora_state_dicts: list[dict[str, torch.Tensor]],
) -> None:
    """
    Verify that all LoRA experts have:
      - identical key sets
      - identical tensor shapes
      - identical dtypes

    This is required for weighted merging.
    """
    if not lora_state_dicts:
        raise ValueError("No LoRA state dicts were loaded.")

    ref_keys = set(lora_state_dicts[0].keys())

    for idx, sd in enumerate(lora_state_dicts[1:], start=1):
        cur_keys = set(sd.keys())
        if cur_keys != ref_keys:
            missing = sorted(ref_keys - cur_keys)
            extra = sorted(cur_keys - ref_keys)
            raise ValueError(
                f"LoRA expert {idx} keys mismatch. "
                f"Missing={missing[:10]} Extra={extra[:10]}"
            )

    for key in sorted(ref_keys):
        ref_shape = lora_state_dicts[0][key].shape
        ref_dtype = lora_state_dicts[0][key].dtype

        for idx, sd in enumerate(lora_state_dicts[1:], start=1):
            if sd[key].shape != ref_shape:
                raise ValueError(
                    f"Architecture mismatch for key={key}: "
                    f"{ref_shape} vs {sd[key].shape} at expert index {idx}"
                )
            if sd[key].dtype != ref_dtype:
                raise ValueError(
                    f"Dtype mismatch for key={key}: "
                    f"{ref_dtype} vs {sd[key].dtype} at expert index {idx}"
                )


def create_template_peft_model(
    model_name_or_path: str,
    template_adapter_path: str,
    torch_dtype: str = "auto",
) -> PeftModel:
    """
    Build a PEFT model using the first expert as a structural template.

    We do not swap whole models during optimization. Instead, we keep one PEFT
    model alive and overwrite its LoRA parameters in-place at each Nevergrad step.
    """
    base_model = load_base_model(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch_dtype,
    )
    peft_model = PeftModel.from_pretrained(base_model, template_adapter_path)
    peft_model.to(DEVICE)
    peft_model.eval()
    return peft_model


def get_lora_parameter_refs(model: nn.Module) -> dict[str, nn.Parameter]:
    """
    Collect references to all live LoRA parameters in the template PEFT model.

    Returns:
        Dict[name -> nn.Parameter]
    """
    refs: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            refs[name] = param

    if not refs:
        raise ValueError("No LoRA parameters found in template PEFT model.")

    return refs


def validate_template_matches_state(
    lora_param_refs: dict[str, nn.Parameter],
    lora_state_dicts: list[dict[str, torch.Tensor]],
) -> None:
    """
    Verify that the template PEFT model's LoRA parameter names/shapes match
    the loaded expert adapter tensors.
    """
    model_keys = set(lora_param_refs.keys())
    state_keys = set(lora_state_dicts[0].keys())

    if model_keys != state_keys:
        missing = sorted(model_keys - state_keys)
        extra = sorted(state_keys - model_keys)
        raise ValueError(
            "Template PEFT model keys do not match expert adapter keys. "
            f"Missing={missing[:10]} Extra={extra[:10]}"
        )

    for key, param in lora_param_refs.items():
        if param.shape != lora_state_dicts[0][key].shape:
            raise ValueError(
                f"Shape mismatch between template model and adapter state for key={key}: "
                f"{tuple(param.shape)} vs {tuple(lora_state_dicts[0][key].shape)}"
            )


def _extract_lengths_for_sorting(dataset: Any) -> list[int]:
    """
    Extract sequence lengths from the dataset in a way that stays compatible
    with the typical Hugging Face dataset interface used by `dataset.py`.

    We prefer column access if available because it is usually cheaper than
    repeated `dataset[i]` indexing from Python.
    """
    # Fast path for Hugging Face-style datasets
    try:
        column = dataset["input_ids"]
        return [len(x) for x in column]
    except Exception:
        pass

    # Fallback path for generic datasets
    try:
        return [len(dataset[i]["input_ids"]) for i in range(len(dataset))]
    except Exception as e:
        raise ValueError(
            "Could not extract sample lengths from dataset. "
            "Expected each sample to contain an 'input_ids' field."
        ) from e


def create_dataloader(
    dataset: Any,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a dataloader with length-sorted batches.

    Sorting by input length reduces padding waste and improves causal LM
    throughput when evaluating many candidate LoRA mixtures.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    lengths = _extract_lengths_for_sorting(dataset)
    sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
    dataset = dataset.select(sorted_indices)

    collator = DataCollatorForSupervisedFinetuning(pad_token_id=tokenizer.pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def weighted_sum_lora(
    lora_state_dicts: list[dict[str, torch.Tensor]], weights: np.ndarray
) -> dict[str, torch.Tensor]:
    """
    Form a weighted linear combination of LoRA tensors:

        merged[key] = sum_i weights[i] * lora_i[key]

    Assumes all expert tensors are already on the target device.
    """
    merged: dict[str, torch.Tensor] = {}
    weights = np.asarray(weights, dtype=np.float32)

    for key in lora_state_dicts[0].keys():
        first = lora_state_dicts[0][key]

        # Allocate directly on the same device/dtype as expert tensors.
        out = torch.zeros_like(first, device="cpu")

        for w, sd in zip(weights.tolist(), lora_state_dicts):
            out.add_(sd[key], alpha=float(w))

        merged[key] = out

    return merged


@torch.no_grad()
def inject_lora_weights_(
    lora_param_refs: dict[str, nn.Parameter],
    merged_lora: dict[str, torch.Tensor],
) -> None:
    """
    Overwrite the live LoRA parameters in the template PEFT model in-place.
    """
    for name, param in lora_param_refs.items():
        tensor = merged_lora[name]
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


@torch.no_grad()
def compute_loss(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()

    total_weighted_loss = 0.0
    total_supervised_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating batches", leave=False):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        supervised_tokens = int(batch["labels"].ne(-100).sum().item())

        if supervised_tokens == 0:
            continue

        total_weighted_loss += float(loss.item()) * supervised_tokens
        total_supervised_tokens += supervised_tokens

    return total_weighted_loss / total_supervised_tokens


def build_optimizer(
    n_experts: int,
    max_steps: int,
    weight_bound: float | None,
) -> ng.optimizers.base.Optimizer:
    """
    Build the Nevergrad optimizer over a real-valued expert weight vector.
    """
    if n_experts <= 0:
        raise ValueError(f"n_experts must be > 0, got {n_experts}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    param = ng.p.Array(shape=(n_experts,))

    if weight_bound is not None:
        param = param.set_bounds(-weight_bound, weight_bound)

    return ng.optimizers.NGOpt(
        parametrization=param,
        budget=max_steps,
    )


# (all previous helper functions remain identical)
# load_lora_state_dicts
# validate_lora_state_dicts
# create_template_peft_model
# get_lora_parameter_refs
# validate_template_matches_state
# create_dataloader
# weighted_sum_lora
# inject_lora_weights_
# compute_loss
# build_optimizer
# etc.


def lorahub_learning(
    config: Any,
) -> tuple[np.ndarray, PeftModel, PreTrainedTokenizerBase, list[dict[str, Any]]]:
    """
    Learn a weighted composition of expert LoRAs that minimizes supervised loss
    on the dataset returned by `mixer.dataset.get_dataset(...)`.

    Expected config fields:
        - model_name_or_path
        - expert_paths
        - train_set_configs
        - batch_size
        - max_steps
        - l1_regularization
        - weight_bound
        - torch_dtype
        - seed
        - num_workers
        - checkpoint_dir
        - resume
        - checkpoint_interval
    """
    set_seed(config.seed)

    history: list[dict[str, Any]] = []

    print("Loading expert LoRA state dicts...")
    lora_state_dicts, inferred_base_model = load_lora_state_dicts(
        config.expert_paths,
        device=DEVICE,
    )
    validate_lora_state_dicts(lora_state_dicts)

    if config.model_name_or_path != inferred_base_model:
        raise ValueError(
            "config.model_name_or_path does not match the experts' base model. "
            f"Config={config.model_name_or_path}, experts={inferred_base_model}"
        )

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name_or_path)

    print("Loading dataset...")
    dataset = get_dataset(
        cfgs=config.train_set_configs,
        tokenizer=tokenizer,
        seed=config.seed,
    )

    dataloader = create_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    print("Building template PEFT model...")
    model = create_template_peft_model(
        model_name_or_path=config.model_name_or_path,
        template_adapter_path=config.expert_paths[0],
        torch_dtype=config.torch_dtype,
    )

    lora_param_refs = get_lora_parameter_refs(model)
    validate_template_matches_state(lora_param_refs, lora_state_dicts)

    n_experts = len(lora_state_dicts)
    step_counter = 0

    checkpoint_dir = config.checkpoint_dir or os.path.join(
        config.output_dir, "checkpoints"
    )

    if config.resume and os.path.exists(
        os.path.join(checkpoint_dir, "optimizer_state.pkl")
    ):
        print("Resuming optimizer from checkpoint...")
        optimizer, step_counter, history = load_checkpoint(checkpoint_dir)
    else:
        optimizer = build_optimizer(
            n_experts=n_experts,
            max_steps=config.max_steps,
            weight_bound=config.weight_bound,
        )

    progress = tqdm(
        total=config.max_steps,
        desc="Optimizing LoRA weights",
        initial=step_counter,
    )

    def score(weights: np.ndarray) -> float:
        nonlocal step_counter

        weights = np.asarray(weights, dtype=np.float32)

        merged = weighted_sum_lora(
            lora_state_dicts=lora_state_dicts,
            weights=weights,
        )
        inject_lora_weights_(lora_param_refs, merged)

        loss = compute_loss(model, dataloader)
        reg = float(np.sum(np.abs(weights)) * config.l1_regularization)
        total = float(loss + reg)

        record = {
            "step": step_counter,
            "loss": float(loss),
            "regularization": reg,
            "total": total,
            "weights": weights.tolist(),
        }

        history.append(record)

        progress.update(1)
        progress.set_postfix({"loss": f"{loss:.4f}", "reg": f"{reg:.4f}"})

        # Periodically save optimizer checkpoint
        if config.checkpoint_interval > 0 and (
            step_counter % config.checkpoint_interval == 0
        ):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                step=step_counter,
                history=history,
            )

        step_counter += 1

        return total

    recommendation = optimizer.minimize(score)

    progress.close()

    best_weights = np.asarray(recommendation.value, dtype=np.float32)

    print("Applying best weights...")

    merged = weighted_sum_lora(
        lora_state_dicts=lora_state_dicts,
        weights=best_weights,
    )

    inject_lora_weights_(lora_param_refs, merged)

    model.eval()

    print("Best weights:", best_weights.tolist())

    return best_weights, model, tokenizer, history
