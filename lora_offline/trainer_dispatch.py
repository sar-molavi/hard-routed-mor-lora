"""Dispatch-aware trainers that slice duplicated batches per rank.

These classes are intended for `accelerator_config={"dispatch_batches": True}`
where the DataLoader runs on rank 0 and the same global batch is broadcast to
all ranks. We slice the global batch evenly so each rank trains on a distinct
shard, keeping effective per-rank batch size and step counts aligned with the
non-dispatch path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.distributed as dist

from .trainer import SequenceLevelOfflineTrainer, TokenLevelOfflineTrainer

logger = logging.getLogger(__name__)


class _DispatchBatchMixin:
    """Mixin that slices the broadcasted batch per rank when dispatching."""

    def _dispatch_enabled(self) -> bool:
        cfg = getattr(self.args, "accelerator_config", None)
        if cfg is None:
            enabled = False
        elif hasattr(cfg, "dispatch_batches"):
            enabled = bool(cfg.dispatch_batches)
        else:
            # Fallback for dicts
            enabled = bool(getattr(cfg, "get", lambda *_: None)("dispatch_batches"))

        # Emit a single warning so we can see the resolved status at runtime.
        if not hasattr(self, "_dispatch_warned"):
            logger.warning("dispatch_batches resolved to %s", enabled)
            self._dispatch_warned = True

        return enabled

    def _select_rank_batch(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not (
            self._dispatch_enabled() and dist.is_available() and dist.is_initialized()
        ):
            return inputs

        world_size = dist.get_world_size()
        if world_size <= 1:
            return inputs

        rank = dist.get_rank()
        batch_size = self._batch_size_from_inputs(inputs)
        if batch_size == 0:
            return inputs

        # Evenly split with first ranks receiving at most one extra example.
        base, extra = divmod(batch_size, world_size)
        start = rank * base + min(rank, extra)
        end = start + base + (1 if rank < extra else 0)

        def _slice(obj):
            if isinstance(obj, torch.Tensor) and obj.size(0) == batch_size:
                return obj[start:end]
            return obj

        return {k: _slice(v) for k, v in inputs.items()}

    def _truncate_padding(self, inputs: dict[str, Any]) -> dict[str, Any]:
        attention_mask = inputs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2:
            return inputs

        seq_len = attention_mask.size(1)
        max_len = int(attention_mask.sum(dim=1).max().item())
        if max_len == seq_len:
            return inputs

        start = seq_len - max_len
        end = seq_len

        def _slice(obj):
            if (
                isinstance(obj, torch.Tensor)
                and obj.dim() >= 2
                and obj.size(1) == seq_len
            ):
                return obj[:, start:end, ...]
            return obj

        return {k: _slice(v) for k, v in inputs.items()}

    def training_step(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        # Slice the shared batch so each rank processes its shard only.
        inputs = self._select_rank_batch(inputs)
        inputs = self._truncate_padding(inputs)
        return super().training_step(model, inputs, num_items_in_batch)


class DispatchSequenceLevelOfflineTrainer(
    _DispatchBatchMixin, SequenceLevelOfflineTrainer
):
    """Sequence-level objective with batch slicing for dispatch mode."""

    # No extra logic; mixin handles slicing.


class DispatchTokenLevelOfflineTrainer(_DispatchBatchMixin, TokenLevelOfflineTrainer):
    """Token-level objective with batch slicing for dispatch mode."""

    # No extra logic; mixin handles slicing.


__all__ = [
    "DispatchSequenceLevelOfflineTrainer",
    "DispatchTokenLevelOfflineTrainer",
]
