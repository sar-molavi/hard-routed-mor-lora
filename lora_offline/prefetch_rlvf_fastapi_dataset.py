"""Prefetch pipeline that talks to a local FastAPI vLLM server."""

from __future__ import annotations

import asyncio
import math
import queue
import threading
from dataclasses import replace, dataclass
from typing import Any, Iterable
from pathlib import Path
import time


import httpx
import numpy as np
from datasets import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .dataset import get_dataset

DATASET_STATE_NAME = "prefetch_state.pt"


@dataclass
class PrefetchConfig:
    # dataset + server
    dataset_name: str
    dataset_path: str
    checkpoints_dir: str | Path
    server_url: str
    model_name: str
    api_key: str | None = None

    # dataset controls
    max_samples: int | None = None

    # request batching
    samples_per_checkpoint: int = 64
    request_batch_size: int = 4
    prefetch_queue_size: int = 16
    request_timeout: float = 120.0

    # generation settings
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float | None = None
    stop_sequences: list[str] | None = None
    logprobs: int = 10
    num_generations: int = 1
    generation_kwargs: dict[str, Any] | None = None

    # shuffling + repetition
    seed: int = 1997
    n_repeat: int = 3

    # checkpoint tracking
    wait_for_new_checkpoint: bool = False
    checkpoint_poll_interval: float = 1.0
    checkpoint_wait_timeout: float | None = 3600
    checkpoint_min_step_delta: int = 0
    allow_missing_initial_checkpoint: bool = True


class CheckpointTracker:
    """Keeps track of the most recent ``checkpoint-*`` directory on disk."""

    _ADAPTER_CONFIG_NAME = "adapter_config.json"
    _ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")

    def __init__(
        self,
        checkpoints_dir: str | Path,
        *,
        wait_for_new: bool = True,
        poll_interval: float = 1.0,
        wait_timeout: float | None = 1800,
        min_step_delta: int = 0,
        allow_missing_initial: bool = True,
    ) -> None:
        self._dir = Path(checkpoints_dir)
        self._current = self._scan_latest()
        self._wait_for_new = wait_for_new
        self._poll_interval = max(poll_interval, 0.1)
        self._wait_timeout = wait_timeout
        self._min_step_delta = max(int(min_step_delta or 0), 0)
        self._allow_missing_initial = allow_missing_initial
        self._initial_checked = False

    @property
    def current(self) -> str | None:
        """Return the cached checkpoint path (or ``None`` when missing)."""
        return self._current

    def refresh(self) -> str | None:
        if self._allow_missing_initial and not self._initial_checked:
            self._current = self._scan_latest()
            self._initial_checked = True
            return self._current

        if not self._wait_for_new:
            self._current = self._scan_latest()
            return self._current

        if self._wait_for_new:
            deadline = time.time() + max(self._wait_timeout, 0.0)
            while time.time() < deadline:
                latest = self._scan_latest()
                if self._meets_min_delta(latest):
                    self._current = latest
                    return self._current

                time.sleep(self._poll_interval)
            return self._current

    def _scan_latest(self) -> str | None:
        if not self._dir.exists():
            return None
        checkpoints = [
            path
            for path in self._dir.glob("checkpoint-*")
            if path.is_dir() and self._is_lora_checkpoint(path)
        ]
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda path: self._extract_step(path) or -1)
        latest = checkpoints[-1]
        return str(latest.resolve())

    def _meets_min_delta(self, latest: str) -> bool:
        latest_step = self._extract_step(latest)
        current_step = self._extract_step(self._current)

        current_step = -1 if current_step is None else current_step
        latest_step = -1 if latest_step is None else latest_step

        return (latest_step - current_step) >= self._min_step_delta

    @staticmethod
    def _extract_step(path: str | Path | None) -> int | None:
        if path is None:
            return None
        name = Path(path).name
        try:
            return int(name.split("checkpoint-")[-1])
        except ValueError:
            return None

    @classmethod
    def _is_lora_checkpoint(cls, path: Path) -> bool:
        config_path = path / cls._ADAPTER_CONFIG_NAME
        if not config_path.is_file():
            return False
        for name in cls._ADAPTER_WEIGHTS:
            if (path / name).is_file():
                return True
        return False


class _FastAPIPrefetcher:
    _SENTINEL = object()

    def __init__(
        self,
        *,
        dataset: Iterable[dict[str, Any]],
        total_samples: int,
        request_batch_size: int,
        prefetch_queue_size: int,
        samples_per_checkpoint: int,
        tracker: CheckpointTracker,
        server_url: str,
        generation_kwargs: dict[str, Any],
        request_timeout: float,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        if request_batch_size < 1:
            raise ValueError("`request_batch_size` must be >= 1.")
        if prefetch_queue_size < 1:
            raise ValueError("`prefetch_queue_size` must be >= 1.")
        if samples_per_checkpoint < 1:
            raise ValueError("`samples_per_checkpoint` must be >= 1.")

        self._dataset_iter = iter(dataset)
        self._remaining = total_samples
        self._request_batch_size = request_batch_size
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_queue_size)
        self._samples_per_checkpoint = samples_per_checkpoint
        self._tracker = tracker
        self._server_url = server_url.rstrip("/")
        self._generation_kwargs = generation_kwargs
        self._request_timeout = request_timeout
        self._tokenizer = tokenizer

        self._stop_event = threading.Event()
        self._shutdown_complete = threading.Event()
        self._error: Exception | None = None

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def get(self) -> dict[str, Any] | None:
        if self._error:
            raise RuntimeError("Prefetcher failed") from self._error

        item = self._queue.get()
        if item is self._SENTINEL:
            self._queue.put(self._SENTINEL)
            if self._error:
                raise RuntimeError("Prefetcher failed") from self._error
            return None
        return item

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._shutdown_complete.wait()
        self._thread.join(timeout=5)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except Exception as exc:
            print(f"Prefetch worker crashed: {exc!r}")
            import traceback

            traceback.print_exc()
            self._error = exc
            self._queue.put(self._SENTINEL)
        finally:
            self._shutdown_complete.set()

    async def _async_main(self) -> None:
        async with httpx.AsyncClient(
            base_url=self._server_url,
            timeout=self._request_timeout,
        ) as client:

            async def _post_ok(path: str, payload: dict[str, Any]) -> None:
                response = await client.post(path, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Server error {response.status_code}: {response.text}"
                    )
                data = response.json()
                if not data.get("ok", False):
                    raise RuntimeError(data.get("error", "Server error"))

            async def _queue_put(record: dict[str, Any]) -> None:
                await asyncio.to_thread(self._queue.put, record)

            async def _send_one(
                example: dict[str, Any], *, current_path: str | None, priority: int
            ) -> dict[str, Any]:
                prompt_token_ids = example["prompt_token_ids"]
                payload = {
                    "prompt_token_ids": prompt_token_ids,
                    "sampling_params": self._generation_kwargs,
                    "lora_path": current_path,
                    "priority": priority,
                }
                response = await client.post("/generate_one", json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Server error {response.status_code}: {response.text}"
                    )
                data = response.json()
                if not data.get("ok", False):
                    raise RuntimeError(data.get("error", "Server error"))
                result = data["result"]
                records = _format_fastapi_batch(
                    batch=[example], results=[result], tokenizer=self._tokenizer
                )
                return records[0]

            quota = max(self._samples_per_checkpoint, 1)
            produced_under_adapter = quota
            current_path: str | None = None
            total_batches = max(
                1, math.ceil(self._remaining / max(self._request_batch_size, 1))
            )
            sent_samples = 0

            try:
                while not self._stop_event.is_set():
                    if self._remaining <= 0:
                        break

                    if produced_under_adapter >= quota:
                        checkpoint_path = self._tracker.refresh()
                        if (
                            checkpoint_path != current_path
                            or produced_under_adapter >= quota
                        ):
                            if current_path:
                                await _post_ok(
                                    "/lora/unregister", {"lora_path": current_path}
                                )
                            current_path = checkpoint_path
                            if current_path:
                                await _post_ok(
                                    "/lora/register", {"lora_path": current_path}
                                )
                            produced_under_adapter = 0

                    target = min(
                        quota - produced_under_adapter,
                        self._remaining,
                    )
                    examples = self._next_examples(target)
                    if not examples:
                        break

                    tasks: list[asyncio.Task[dict[str, Any]]] = []
                    for example in examples:
                        priority = total_batches - (
                            sent_samples // self._request_batch_size
                        )
                        tasks.append(
                            asyncio.create_task(
                                _send_one(
                                    example,
                                    current_path=current_path,
                                    priority=priority,
                                )
                            )
                        )
                        sent_samples += 1
                    try:
                        for future in asyncio.as_completed(tasks):
                            record = await future
                            await _queue_put(record)
                            self._remaining -= 1
                            produced_under_adapter += 1
                    except Exception:
                        await asyncio.gather(
                            *tasks,
                            return_exceptions=True,
                        )
                        raise
            finally:
                if current_path:
                    await _post_ok("/lora/unregister", {"lora_path": current_path})
                self._queue.put(self._SENTINEL)

    def _next_examples(self, limit: int) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while len(batch) < limit:
            try:
                example = next(self._dataset_iter)
            except StopIteration:
                break
            prompt_token_ids = example.get("prompt_token_ids")
            if not prompt_token_ids:
                continue
            example = dict(example)
            batch.append(example)
        return batch


def _format_fastapi_batch(
    *,
    batch: list[dict[str, Any]],
    results: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, example in enumerate(batch):
        result = results[idx] if idx < len(results) else {}
        token_ids = result["token_ids"]
        generated_text = [
            tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for ids in token_ids
        ]
        output.append(
            {
                "prompt": example["prompt"],
                "input_token_ids": example["prompt_token_ids"],
                "generated_token_ids": token_ids,
                "ground_truth": example["ground_truth"],
                "generated_text": generated_text,
                "token_logprobs": result["token_logprobs"],
                "finish_reason": result["finish_reason"],
            }
        )
    return output


class FastAPIPrefetchIterableDataset(IterableDataset):
    def __init__(self, config: PrefetchConfig):
        self.config = config
        self.seed = (
            config.seed if config.seed is not None else np.random.randint(0, 2**32 - 1)
        )
        self.emitted = 0
        self.generated_samples = 0
        self.generation_kwargs = self._build_generation_kwargs()
        self._init_dataset()

    def state_dict(self) -> dict:
        return {"seed": self.seed, "emitted": self.emitted}

    def load_state_dict(self, state_dict: dict):
        self.seed = state_dict["seed"]
        self.emitted = state_dict["emitted"]

    def _build_generation_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        settings: dict[str, Any] = {
            "n": cfg.num_generations,
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "logprobs": max(int(cfg.logprobs or 0), 1),
            "presence_penalty": cfg.presence_penalty,
            "frequency_penalty": cfg.frequency_penalty,
            "include_stop_str_in_output": True,
            "spaces_between_special_tokens": False,
            "detokenize": False,
            **(cfg.generation_kwargs or {}),
        }
        if cfg.top_k is not None:
            settings["top_k"] = cfg.top_k
        if cfg.repetition_penalty is not None:
            settings["repetition_penalty"] = cfg.repetition_penalty
        if cfg.stop_sequences:
            settings["stop"] = cfg.stop_sequences
        return settings

    def _init_dataset(self):
        cfg = self.config
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer
        dataset = get_dataset(
            dataset_name=cfg.dataset_name,
            dataset_path=cfg.dataset_path,
            tokenizer=tokenizer,
        )

        def _tokenize_prompt(example: dict) -> dict:
            prompt_token_ids = tokenizer(
                example["prompt"], add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
            return {"prompt_token_ids": prompt_token_ids}

        self.dataset = dataset.map(_tokenize_prompt, batched=True)

    def _build_repeated_shuffled_indices(self, length: int):
        cfg = self.config
        all_idx = []
        for j in range(cfg.n_repeat):
            rng = np.random.default_rng(self.seed + j)
            all_idx.append(rng.permutation(length))
        return np.concatenate(all_idx)

    def __iter__(self):
        cfg = self.config
        dataset = self.dataset

        length = len(dataset)
        indices = self._build_repeated_shuffled_indices(length)
        dataset = dataset.select(indices.tolist())

        if cfg.max_samples is not None:
            dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

        if self.emitted > 0:
            dataset = dataset.skip(self.emitted)

        dataset = dataset.with_format("python")
        total = len(dataset)

        tracker = CheckpointTracker(
            cfg.checkpoints_dir,
            wait_for_new=cfg.wait_for_new_checkpoint,
            poll_interval=cfg.checkpoint_poll_interval,
            wait_timeout=cfg.checkpoint_wait_timeout,
            min_step_delta=cfg.checkpoint_min_step_delta,
            allow_missing_initial=cfg.allow_missing_initial_checkpoint,
        )

        prefetcher = _FastAPIPrefetcher(
            dataset=dataset,
            total_samples=total,
            request_batch_size=cfg.request_batch_size,
            prefetch_queue_size=cfg.prefetch_queue_size,
            samples_per_checkpoint=cfg.samples_per_checkpoint,
            tracker=tracker,
            server_url=cfg.server_url,
            generation_kwargs=self.generation_kwargs,
            request_timeout=cfg.request_timeout,
            tokenizer=self._tokenizer,
        )

        try:
            local = 0
            while local < total:
                item = prefetcher.get()
                if item is None:
                    break
                local += 1
                yield item
                self.emitted += 1
                self.generated_samples += 1
        finally:
            prefetcher.stop()


class AsyncFastAPIPrefetchPipeline:
    def __init__(self, **kwargs):
        self._config = PrefetchConfig(**kwargs)

    @property
    def config(self) -> PrefetchConfig:
        return self._config

    def build_dataset(
        self, *, max_samples: int | None = None
    ) -> FastAPIPrefetchIterableDataset:
        cfg = self._config
        if max_samples is not None and max_samples != cfg.max_samples:
            cfg = replace(cfg, max_samples=max_samples)
        return FastAPIPrefetchIterableDataset(cfg)
