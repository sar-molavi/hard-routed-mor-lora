"""Utility helpers for prompt parsing and answer extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import (
    PretrainedConfig,
)
import torch

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


def extract_last_after_hashes(text: str) -> str | None:
    """
    Return substring following the last occurrence of ``####``.
    """
    matches = re.findall(r"####\s*(.+)", text)
    return matches[-1] if matches else None


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
