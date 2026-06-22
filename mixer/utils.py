"""Utility helpers for the LoRA-Mixer training stack."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Iterable
import json
import re

from transformers import PretrainedConfig
import torch
import torch.nn as nn
from peft import PeftModel

LORA_MIXER_WEIGHTS_NAME = "lora_mixer.pth"
PRINT_LOAD_INFO = False


def load_checkpoint_path(path: Path | str) -> str | None:
    """
    Return the checkpoint directory with the highest numeric suffix inside ``path``.
    For example, among 'checkpoint-100', 'checkpoint-20', it returns 'checkpoint-100'.
    """
    path = Path(path)
    if not path.is_dir():
        return None

    checkpoint_dirs = []
    for candidate in path.iterdir():
        if candidate.is_dir() and candidate.name.startswith("checkpoint-"):
            # Extract the numeric suffix safely
            suffix = candidate.name.replace("checkpoint-", "")
            if suffix.isdigit():
                checkpoint_dirs.append((int(suffix), candidate))

    if not checkpoint_dirs:
        return None

    # Select the entry with the highest numeric suffix
    _, latest = max(checkpoint_dirs, key=lambda x: x[0])
    return str(latest)


def print_trainable_parameters(model: nn.Module):
    """
    Calculates and prints the number of trainable parameters in a model.

    Args:
        model: The model to inspect.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params if total_params else 0.0
    print(f"Trainable parameters: {trainable_params} / {total_params} ({pct:.2f}%)")


def save_lora_mixer_weights(model: nn.Module, save_path: str | Path) -> Path:
    """
    Persist only the router modules and LoRA parameter tensors.

    Returns the path to the written checkpoint file.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model = unwrap_model(model)
    moe_layers = model.moe_layers
    if not isinstance(moe_layers, nn.ModuleDict):
        raise ValueError("Model does not expose `moe_layers` as an nn.ModuleDict.")

    enable_lora_attn = bool(model.enable_lora_attn)
    if enable_lora_attn:
        if not isinstance(model.wrapped_model, PeftModel):
            raise ValueError("Model is expected to be a PeftModel instance.")
        model.wrapped_model.save_pretrained(save_path)

    weights_path = save_path / LORA_MIXER_WEIGHTS_NAME

    payload: dict[str, dict] = {}
    parts: list[str] = []
    if not model.freeze_experts:
        lora_state = {
            name: param.detach().cpu()
            for name, param in moe_layers.state_dict().items()
            if name.endswith(".lora_A") or name.endswith(".lora_B")
        }
        payload["lora_params"] = lora_state
        parts.append("LoRA")

    if not model.freeze_router:
        router_state: dict[str, dict[str, torch.Tensor]] | dict[str, torch.Tensor]
        if getattr(model, "router_shared_across_layers", False):
            shared_router = getattr(model, "shared_router", None)
            if shared_router is None:
                raise ValueError("Model is expected to expose `shared_router`.")
            router_state = {"shared": shared_router.state_dict()}
            payload["router_shared"] = True
        else:
            router_layers = getattr(model, "router_layers", None)
            if not isinstance(router_layers, nn.ModuleDict):
                raise ValueError("Model is expected to expose `router_layers`.")
            router_state = router_layers.state_dict()
            payload["router_shared"] = False
        payload["routers"] = router_state
        parts.append("Router")

    torch.save(payload, weights_path)

    label = " + ".join(parts) if parts else "No trainable"
    msg = f"[INFO] {label} parameters saved to {weights_path}"
    print(msg)

    return weights_path


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying model when wrapped with DDP/FSDP/DataParallel.
    """
    while hasattr(model, "module"):
        model = model.module  # type: ignore[attr-defined]
    return model


def clone_lora_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Snapshot the current LoRA parameters for later L2 regularization.
    """
    model = unwrap_model(model)

    moe_layers = model.moe_layers
    if not isinstance(moe_layers, nn.ModuleDict):
        return {}

    return {
        name: param.clone().detach()
        for name, param in moe_layers.named_parameters()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }


def resolve_dtype(*, bf16: bool, fp16: bool) -> torch.dtype:
    """
    Determine the model dtype requested by the configuration.
    """
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def ensure_list(value: str | Iterable[str] | None) -> list[str]:
    """
    Normalize optional string / iterable inputs into a list of strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def load_lora_mixer_weights(
    model: nn.Module, checkpoint_dir: str | Path, *, strict: bool = False
) -> None:
    """
    Load router + LoRA parameters from disk into an existing LoRA-Mixer model.

    Args:
        model: The model instance (or distributed wrapper) receiving the weights.
        checkpoint_path: Directory or file pointing to `lora_mixer.pth`.
        strict: If True, enforce that both router and LoRA keys are present.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_dir():
        checkpoint_path = checkpoint_dir / LORA_MIXER_WEIGHTS_NAME
    else:
        raise ValueError("checkpoint_dir must be a directory.")

    payload = torch.load(checkpoint_path, map_location="cpu")

    model = unwrap_model(model)
    moe_layers = model.moe_layers

    loaded_parts: list[str] = []

    if "routers" in payload:
        router_payload = payload["routers"]
        router_shared = payload.get("router_shared")
        if router_shared is None and isinstance(router_payload, dict):
            if "shared" in router_payload and not any(
                key.startswith("layer") for key in router_payload.keys()
            ):
                router_shared = True
        if router_shared:
            shared_router = getattr(model, "shared_router", None)
            if shared_router is None:
                raise ValueError("Model is expected to expose `shared_router`.")
            shared_state = router_payload.get("shared", router_payload)
            shared_router.load_state_dict(shared_state, strict=strict)
        else:
            router_layers = getattr(model, "router_layers", None)
            if not isinstance(router_layers, nn.ModuleDict):
                raise ValueError("Model is expected to expose `router_layers`.")
            router_layers.load_state_dict(router_payload, strict=strict)
        loaded_parts.append("Router")
    elif not model.freeze_router:
        if strict:
            raise KeyError("Checkpoint missing 'routers' state.")

    if "lora_params" in payload:
        moe_layers.load_state_dict(payload["lora_params"], strict=strict)
        loaded_parts.append("LoRA")
    elif not model.freeze_experts:
        if strict:
            raise KeyError("Checkpoint missing 'lora_params' state.")

    enable_lora_attn = bool(model.enable_lora_attn)
    if enable_lora_attn:
        if not isinstance(model.wrapped_model, PeftModel):
            raise ValueError("Model is expected to be a PeftModel instance.")
        model.wrapped_model.load_adapter(
            str(checkpoint_dir),
            adapter_name="default",
            is_trainable=True,
        )
        loaded_parts.append("Attention LoRA")

    if PRINT_LOAD_INFO:
        label = " + ".join(loaded_parts) if loaded_parts else "No parameters"
        print(f"[INFO] Loaded {label} from {checkpoint_path}")


def get_max_length(model: torch.nn.Module, config: PretrainedConfig = None) -> int:
    """
    Inspect model configuration to determine the maximum supported sequence length.

    Args:
        model (torch.nn.Module): Loaded base model.

    Returns:
        int: Maximum token length supported by the model configuration.
    """
    if model is None and config is None:
        raise Exception("Either model or config must be provided.")

    if model is not None:
        config = model.config

    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(config, length_setting, None)
        if max_length:
            break
    return max_length


def _cleanup_processes() -> None:
    """Best-effort cleanup for distributed state and child processes."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            abort = getattr(dist, "abort", None)
            if callable(abort):
                abort()
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        children = multiprocessing.active_children()
        for child in children:
            child.terminate()
        for child in children:
            child.join(timeout=1)
    except Exception:
        pass


def install_exit_handlers() -> None:
    """Install signal/exception handlers to ensure child processes exit."""
    if getattr(install_exit_handlers, "_installed", False):
        return
    install_exit_handlers._installed = True  # type: ignore[attr-defined]

    cleaned = {"done": False}

    def _cleanup_once() -> None:
        if cleaned["done"]:
            return
        cleaned["done"] = True
        _cleanup_processes()

    def _signal_handler(signum, _frame) -> None:
        _cleanup_once()
        raise SystemExit(128 + signum)

    def _excepthook(exctype, value, tb) -> None:
        _cleanup_once()
        sys.__excepthook__(exctype, value, tb)

    original_thread_excepthook = getattr(threading, "excepthook", None)

    def _thread_excepthook(args) -> None:
        _cleanup_once()
        if callable(original_thread_excepthook):
            original_thread_excepthook(args)
        os._exit(1)

    for sig_name in ("SIGINT", "SIGTERM", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _signal_handler)

    sys.excepthook = _excepthook
    if original_thread_excepthook is not None:
        threading.excepthook = _thread_excepthook
    atexit.register(_cleanup_once)

def extract_last_json(text):
    """
    Extract the last valid JSON object or array from a text string.

    Args:
        text (str): The text containing JSON data

    Returns:
        dict/list: The parsed JSON object or array, or None if no valid JSON found
    """
    # Find all potential JSON objects {...} and arrays [...]
    # Using a regex to find balanced braces/brackets
    json_candidates = []

    # Find all positions where JSON might start
    for match in re.finditer(r"[{\[]", text):
        start = match.start()
        # Try to parse JSON from this position to the end
        for end in range(len(text), start, -1):
            substring = text[start:end]
            try:
                parsed = json.loads(substring)
                json_candidates.append((start, end, parsed))
                break  # Found valid JSON starting at this position
            except (json.JSONDecodeError, ValueError):
                continue

    # Return the last (rightmost) valid JSON found
    if json_candidates:
        # Sort by start position and return the last one
        json_candidates.sort(key=lambda x: x[0])
        return json_candidates[-1][2]

    return None
