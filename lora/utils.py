import re
import json

import numpy as np
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PretrainedConfig,
)
import torch


class GPUMemoryTrackerCallback(TrainerCallback):
    """
    A Hugging Face TrainerCallback that logs GPU memory usage.

    This callback hooks into the training loop and prints detailed GPU memory
    statistics (allocated and reserved) every `log_every_n_steps`. This is
    particularly useful for:
    - Monitoring memory consumption to prevent out-of-memory errors.
    - Optimizing `per_device_train_batch_size` and `gradient_accumulation_steps`.
    - Detecting potential memory leaks during long training runs.
    """

    def __init__(self, log_every_n_steps: int = 10):
        """
        Initializes the callback.

        Args:
            log_every_n_steps (int, optional):
                The frequency of logging, in training steps. Defaults to 10.
        """
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event hook called by the Trainer at the end of each training step.

        This method checks if the current step is a logging step and, if so,
        prints the memory usage for all available CUDA devices.
        """
        # Log only every N steps to avoid cluttering the output.
        # The `is_world_process_zero` check ensures that this logging is only
        # performed by the main process in a multi-GPU (DDP) setup.
        if (
            state.global_step > 0
            and state.global_step % self.log_every_n_steps == 0
            and state.is_world_process_zero
        ):
            print(f"\n--- GPU Memory Stats at Step {state.global_step} ---")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                print(
                    f"GPU {i}: Allocated={allocated:.2f} MB | Reserved={reserved:.2f} MB"
                )
            print("------------------------------------")


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


def _extract_gsm8k_answer(text: str) -> str:
    """Extract the final GSM8K answer after the last '####' marker."""
    matches = re.findall(r"####\s*(.+)", text)
    if not matches:
        return ""
    return matches[-1].strip().lower()


def _extract_gsm8k_reasoning(text: str) -> str:
    """
    Extract the reasoning portion of a GSM8K solution.
    This is everything before the final '####' marker.
    """
    # Split on the last occurrence of ####
    parts = re.split(r"####\s*", text)
    if len(parts) <= 1:
        return ""

    reasoning = parts[0].strip()

    # Optional: remove trailing whitespace or stray newlines
    return reasoning

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
