"""
Distributed-friendly wrapper for the async prefetching dataset.

The classes here allow only rank 0 to talk to the vLLM/OpenAI server while the
other Distributed Data Parallel workers receive already generated samples. This
avoids redundant remote calls when Hugging Face's Trainer instantiates the
dataset on every rank.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from .prefetch_rlvf_fastapi_dataset import (
    AsyncFastAPIPrefetchPipeline,
    DATASET_STATE_NAME,
    FastAPIPrefetchIterableDataset,
)
from .utils import load_checkpoint_path


class DistributedPrefetchIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], AsyncFastAPIPrefetchPipeline],
        max_samples: int | None = None,
        output_dir: str = ".",
    ) -> None:
        """
        Args:
            pipeline_factory: Callable that lazily constructs an
                :class:`AsyncOpenAIPrefetchPipeline`. Only executed on rank 0.
            max_samples: Optional cap on the total number of RL samples.
            output_dir: Training output directory containing HF checkpoints.
        """
        super().__init__()
        self._pipeline_factory = pipeline_factory
        self._max_samples = max_samples
        self._output_dir = output_dir
        self._last_dataset: FastAPIPrefetchIterableDataset | None = None

    def _load_dataset_state(self, dataset: FastAPIPrefetchIterableDataset) -> None:
        # Apply any pending state passed via `load_state_dict` first.
        latest = load_checkpoint_path(self._output_dir)
        if latest is None:
            return

        path = Path(latest) / DATASET_STATE_NAME
        if not path.is_file():
            return

        state = torch.load(path, map_location="cpu")
        dataset.load_state_dict(state)

    def state_dict(self):
        """
        Allow Trainer to checkpoint dataset progress by delegating to the
        underlying PrefetchIterableDataset built on rank 0.
        """
        if self._last_dataset is not None:
            return self._last_dataset.state_dict()

    def __iter__(self):
        if not dist.is_available() or not dist.is_initialized():
            pipeline = self._pipeline_factory()
            dataset = pipeline.build_dataset(max_samples=self._max_samples)
            self._last_dataset = dataset
            self._load_dataset_state(dataset)
            return iter(dataset)

        rank = dist.get_rank()

        # Only rank 0 fetches data
        if rank == 0:
            pipeline = self._pipeline_factory()
            dataset = pipeline.build_dataset(max_samples=self._max_samples)
            self._last_dataset = dataset
            self._load_dataset_state(dataset)
            return iter(dataset)
        else:
            # Other ranks return empty iterator
            return iter([])


def build_distributed_prefetch_dataset(
    pipeline: AsyncFastAPIPrefetchPipeline,
    *,
    max_samples: int | None = None,
    output_dir: str = ".",
) -> DistributedPrefetchIterableDataset:
    """
    Convenience helper that wraps the given pipeline into the distributed dataset.

    Args:
        pipeline: Configured AsyncFastAPIPrefetchPipeline.
        max_samples: Optional cap forwarded to dataset builder.
    """

    def _factory() -> AsyncFastAPIPrefetchPipeline:
        return pipeline

    return DistributedPrefetchIterableDataset(
        pipeline_factory=_factory,
        max_samples=max_samples,
        output_dir=output_dir,
    )
