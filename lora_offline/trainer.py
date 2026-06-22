"""Offline RL trainers with pluggable objectives."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


@dataclass
class OfflineBatchStats:
    new_token_logprobs: torch.Tensor
    old_token_logprobs: torch.Tensor
    mask: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    stop_flags: torch.Tensor
    lengths: torch.Tensor


class OfflineTrainerBase(Trainer, ABC):
    """
    Base class that extracts token-level log-probabilities and defers the
    objective computation to subclasses.
    """

    def __init__(
        self,
        *args,
        normalize_advantages: bool = False,
        dataset_state_name="prefetch_state.pt",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.normalize_advantages = normalize_advantages
        self.dataset_state_name = dataset_state_name

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Override HF's training_step to split oversized batches into smaller
        micro-batches, preserving gradient semantics even when the collator
        expands each dataset sample into multiple completions.
        """
        # Set the model to training mode.
        model.train()
        # Prepare inputs by moving them to the correct device (e.g., GPU).
        inputs = self._prepare_inputs(inputs)

        # Determine the size of the batch from the input tensors.
        batch_size = self._batch_size_from_inputs(inputs)
        # If the batch is empty, return a zero loss.
        if batch_size <= 0:
            return torch.tensor(0.0, device=self.args.device)

        # Define the size of each micro-batch (chunk).
        chunk_size = max(self.args.per_device_train_batch_size, 1)
        # If the batch is not larger than a single chunk, process it normally using the parent's training_step.
        if batch_size <= chunk_size:
            return super().training_step(model, inputs, num_items_in_batch)

        # Initialize variables for accumulating loss across chunks.
        total_chunks = 0
        total_loss = None
        # Iterate over the batch in chunks.
        for chunk in self._chunk_inputs(inputs, chunk_size):
            # Get the actual size of the current chunk.
            chunk_batch = self._batch_size_from_inputs(chunk)
            # Skip empty chunks.
            if chunk_batch == 0:
                continue
            # Use the context manager for loss computation (e.g., for mixed precision).
            with self.compute_loss_context_manager():
                # Compute the loss for the current chunk.
                loss = self.compute_loss(model, chunk)
            # Scale the loss. The loss from compute_loss is an average over the chunk.
            # We scale it by the chunk's proportion of the total batch to ensure
            # that summing the scaled losses is equivalent to the loss of the full batch.
            loss = loss * (chunk_batch / batch_size)
            # If using gradient accumulation, further scale the loss.
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            # Perform backpropagation.
            self.accelerator.backward(loss)
            # Detach the loss from the computation graph to prevent memory leaks when accumulating.
            detached = loss.detach()
            # Accumulate the detached loss.
            total_loss = detached if total_loss is None else total_loss + detached
            total_chunks += 1

        # If no chunks were processed, return a zero loss.
        if total_chunks == 0:
            return torch.tensor(0.0, device=self.args.device)
        # Return the average loss over all chunks.
        return total_loss / total_chunks

    def _batch_size_from_inputs(self, inputs: dict[str, Any]) -> int:
        # A helper function to find the batch size from the input dictionary.
        # It checks common tensor keys.
        for key in ("input_ids", "attention_mask", "rewards"):
            tensor = inputs.get(key)
            if isinstance(tensor, torch.Tensor):
                # The batch size is the first dimension of the tensor.
                return tensor.size(0)
        # As a fallback, check any tensor in the input values.
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        # If no tensors are found, the batch size is 0.
        return 0

    def _chunk_inputs(
        self, inputs: dict[str, Any], chunk_size: int
    ) -> list[dict[str, Any]]:
        # A helper function to split a batch of inputs into a list of smaller chunks.
        batch_size = self._batch_size_from_inputs(inputs)
        # If the batch is already small enough, return it as a single-element list.
        if batch_size == 0 or batch_size <= chunk_size:
            return [inputs]

        chunks: list[dict[str, Any]] = []
        # Iterate through the batch with a step of chunk_size.
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk: dict[str, Any] = {}
            # For each key-value pair in the inputs...
            for key, value in inputs.items():
                # If the value is a tensor with a batch dimension, slice it.
                if isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                    chunk[key] = value[start:end]
                else:
                    # Otherwise, copy the value as is (e.g., for non-tensor data or metadata).
                    chunk[key] = value
            chunks.append(chunk)
        return chunks

    def compute_loss(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # Prepare inputs specifically for the model's forward pass.
        model_inputs = {
            key: inputs[key] for key in ("input_ids", "attention_mask") if key in inputs
        }
        # Perform the forward pass. `use_cache=False` is important during training.
        # `outputs.logits` shape: (batch_size, seq_len, vocab_size)
        outputs = model(**model_inputs, use_cache=False)
        # Build a structured object with all necessary stats for loss computation.
        stats = self._build_batch_stats(outputs.logits, inputs)
        # Defer the actual objective computation to a subclass.
        loss, metrics = self._compute_objective(stats)
        # Build a dictionary of logs for monitoring and debugging.
        logs = self._build_logs(stats, metrics)
        # Log the computed metrics.
        self.log({k: float(v.detach().mean()) for k, v in logs.items()})
        # Return loss and optionally the model outputs.
        if return_outputs:
            return loss, outputs
        return loss

    def _build_batch_stats(
        self, logits: torch.Tensor, inputs: dict[str, Any]
    ) -> OfflineBatchStats:
        # `logits` shape: (batch_size, seq_len, vocab_size)
        dtype = logits.dtype
        # `input_ids` shape: (batch_size, seq_len)
        input_ids = inputs["input_ids"]
        # `attention_mask` shape: (batch_size, seq_len)
        attention_mask = inputs["attention_mask"]
        # `completion_mask` shape: (batch_size, seq_len). This masks the completion part of the sequence.
        completion_mask = inputs["completion_mask"].to(dtype=attention_mask.dtype)
        # `behavior_logprobs` shape: (batch_size, seq_len). Log-probs from the policy that generated the data.
        behavior_logprobs = torch.nan_to_num(
            inputs["token_logprobs"].to(dtype=dtype), nan=0.0
        )

        # Shift logits and labels for next-token prediction.
        # `shift_logits` shape: (batch_size, seq_len - 1, vocab_size)
        shift_logits = logits[:, :-1, :]
        # `shift_labels` shape: (batch_size, seq_len - 1)
        shift_labels = input_ids[:, 1:]
        # Also shift the completion mask and combine with attention mask.
        # `shift_completion_mask` shape: (batch_size, seq_len - 1)
        shift_completion_mask = completion_mask[:, 1:] * attention_mask[:, 1:]

        # Compute log probabilities from the current model's logits.
        # `log_probs` shape: (batch_size, seq_len - 1, vocab_size)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather the log-probs of the actual next tokens.
        # `new_token_logprobs` shape: (batch_size, seq_len - 1)
        new_token_logprobs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        # Get the log-probs from the behavior policy (shifted).
        # `old_token_logprobs` shape: (batch_size, seq_len - 1)
        old_token_logprobs = behavior_logprobs[:, 1:]

        # Create the final mask for loss computation, focusing only on completion tokens.
        # `mask` shape: (batch_size, seq_len - 1)
        mask = shift_completion_mask.to(dtype=new_token_logprobs.dtype)
        # Apply the mask to the new and old log-probs.
        # `new_token_logprobs` shape: (batch_size, seq_len - 1)
        new_token_logprobs = new_token_logprobs * mask
        # `old_token_logprobs` shape: (batch_size, seq_len - 1)
        old_token_logprobs = torch.nan_to_num(old_token_logprobs, nan=0.0) * mask

        # `advantages` shape: (batch_size,). This is typically a per-sequence value.
        advantages = inputs["advantages"].to(dtype=dtype)
        # Optionally normalize advantages for training stability.
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        # Package all computed stats into a dataclass for easy access.
        stats = OfflineBatchStats(
            # `new_token_logprobs` shape: (batch_size, seq_len - 1)
            new_token_logprobs=new_token_logprobs,
            # `old_token_logprobs` shape: (batch_size, seq_len - 1)
            old_token_logprobs=old_token_logprobs,
            # `mask` shape: (batch_size, seq_len - 1)
            mask=mask,
            # `rewards` shape: (batch_size,)
            rewards=inputs["rewards"].to(dtype=dtype),
            # `advantages` shape: (batch_size,)
            advantages=advantages,
            # `stop_flags` shape: (batch_size,)
            stop_flags=inputs["is_stopped"].float(),
            # `lengths` shape: (batch_size,). Number of completion tokens per sequence.
            lengths=mask.sum(dim=1).clamp(min=1.0),
        )
        return stats

    def _build_logs(
        self, stats: OfflineBatchStats, extra: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Compute standard deviation of rewards for logging.
        reward_std = (
            stats.rewards.std(unbiased=False)
            if stats.rewards.numel() > 1
            else torch.zeros_like(stats.rewards[:1])
        )
        # Create a dictionary of standard metrics to log.
        logs: Dict[str, torch.Tensor] = {
            "reward_mean": stats.rewards.mean(),
            "reward_std": reward_std,
            "advantage_mean": stats.advantages.mean(),
            "completion_length": stats.lengths.mean(),
            "stop_rate": stats.stop_flags.mean(),
        }
        # Add any extra metrics from the objective function.
        logs.update(extra)
        return logs

    @abstractmethod
    def _compute_objective(
        self, stats: OfflineBatchStats
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return the scalar loss and auxiliary metrics for logging.
        This method must be implemented by subclasses.
        """

    # -----------------------------------------------------
    # Checkpoint integration for stateful datasets
    # -----------------------------------------------------
    def _save_checkpoint(self, model, trial):  # type: ignore[override]
        # Get the directory for the current run.
        run_dir = self._get_output_dir(trial=trial)
        # Define the folder for this specific checkpoint.
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(run_dir, checkpoint_folder)
        # Call the parent's save checkpoint method.
        super()._save_checkpoint(model, trial)
        # Also save the state of the dataset.
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self._save_dataset_state(output_dir)
        # NOTE: Do not call a barrier here. In practice, Transformers may call
        # _save_checkpoint only on the main process. A barrier would then hang
        # multi-GPU runs because the other ranks never enter it.

    def _save_dataset_state(self, output_dir: str) -> None:
        # Only save if it's the main process.
        if not self.args.should_save:
            return
        # Get the training dataset.
        dataset = getattr(self, "train_dataset", None)
        if dataset is None:
            return
        # Check if the dataset has a `state_dict` method.
        state_fn = getattr(dataset, "state_dict", None)
        if not callable(state_fn):
            return
        # Get the state from the dataset.
        state = state_fn()
        if state is None:
            return
        # Define the path and save the state.
        path = os.path.join(output_dir, self.dataset_state_name)
        torch.save(state, path)


class SequenceLevelOfflineTrainer(OfflineTrainerBase):
    """Sequence-level importance sampling objective."""

    def __init__(
        self,
        *args,
        ratio_clip: float | None = 0.2,
        ratio_clip_high: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ratio_clip = ratio_clip
        self.ratio_clip_high = ratio_clip_high

    def _compute_objective(
        self, stats: OfflineBatchStats
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_new = stats.new_token_logprobs.sum(dim=1)
        seq_old = stats.old_token_logprobs.sum(dim=1)
        log_ratio = seq_new - seq_old
        ratio = torch.exp(log_ratio)
        min_val = 1.0 - self.ratio_clip if self.ratio_clip is not None else None
        max_delta = (
            self.ratio_clip_high
            if self.ratio_clip_high is not None
            else self.ratio_clip
        )
        max_val = 1.0 + max_delta if max_delta is not None else None
        if min_val is not None or max_val is not None:
            ratio = torch.clamp(ratio, min=min_val, max=max_val)
        loss = -(ratio * stats.advantages).mean()
        metrics = {
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(unbiased=False),
            "log_ratio": log_ratio.mean(),
        }
        return loss, metrics


class TokenLevelOfflineTrainer(OfflineTrainerBase):
    """Token-level importance sampling objective."""

    def __init__(
        self,
        *args,
        ratio_clip: float | None = 0.2,
        ratio_clip_high: float | None = None,
        global_token_normalization: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ratio_clip = ratio_clip
        self.ratio_clip_high = ratio_clip_high
        self.global_token_normalization = global_token_normalization

    def _compute_objective(
        self, stats: OfflineBatchStats
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        token_advantages = stats.advantages.unsqueeze(1) * stats.mask
        log_ratio = (stats.new_token_logprobs - stats.old_token_logprobs) * stats.mask
        ratio = torch.exp(log_ratio)
        min_val = 1.0 - self.ratio_clip if self.ratio_clip is not None else None
        max_delta = (
            self.ratio_clip_high
            if self.ratio_clip_high is not None
            else self.ratio_clip
        )
        max_val = 1.0 + max_delta if max_delta is not None else None
        if min_val is not None or max_val is not None:
            ratio = torch.clamp(ratio, min=min_val, max=max_val)
        losses = -(ratio * token_advantages)
        if self.global_token_normalization:
            total_tokens = stats.mask.sum().clamp(min=1.0)
            loss = losses.sum() / total_tokens
        else:
            loss = (losses.sum(dim=1) / stats.lengths).mean()
        valid = ratio[stats.mask.bool()]
        metrics = {
            "ratio_mean": valid.mean() if valid.numel() > 0 else ratio.new_tensor(0.0),
            "ratio_std": (
                valid.std(unbiased=False)
                if valid.numel() > 1
                else ratio.new_tensor(0.0)
            ),
        }
        return loss, metrics


__all__ = [
    "OfflineTrainerBase",
    "SequenceLevelOfflineTrainer",
    "TokenLevelOfflineTrainer",
]
