"""FastAPI server that hosts an AsyncLLMEngine for local IPC-style HTTP."""

from __future__ import annotations

import os
import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput


@dataclass
class _RegisteredLoRA:
    request: LoRARequest
    checkpoint_path: str
    signature: str | None


def _checkpoint_signature(path: str | None) -> str | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    return str(resolved)


class GenerateRequest(BaseModel):
    prompt_token_ids: list[list[int]]
    sampling_params: dict[str, Any]
    lora_path: str | None = None
    priority: int = 0


class LoraManageRequest(BaseModel):
    lora_path: str | None = None


class GenerateSingleRequest(BaseModel):
    prompt_token_ids: list[int]
    sampling_params: dict[str, Any]
    lora_path: str | None = None
    priority: int = 0


class CompletionResult(BaseModel):
    token_ids: list[list[int]]
    token_logprobs: list[list[float]]
    finish_reason: list[str]


class GenerateResponse(BaseModel):
    ok: bool
    results: list[CompletionResult]
    error: str | None = None


class GenerateSingleResponse(BaseModel):
    ok: bool
    result: CompletionResult
    error: str | None = None


def debug_trace(message: str, **fields: Any) -> None:
    """
    Lightweight debug logger gated by DEBUG_FASTAPI=1.
    """
    if os.environ.get("DEBUG_FASTAPI", "").lower() not in {"1", "true", "yes"}:
        return
    suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
    if suffix:
        print(f"[DEBUG_FASTAPI] {message} {suffix}")
    else:
        print(f"[DEBUG_FASTAPI] {message}")


async def _generate_once(
    *,
    engine: AsyncLLMEngine,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None,
    priority: int,
) -> RequestOutput:
    request_id = uuid.uuid4().hex
    final: RequestOutput | None = None
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    async for output in engine.generate(
        prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=lora_request,
        priority=priority,
    ):
        final = output

    if final is None:
        raise RuntimeError("Engine returned no output")
    return final


async def _generate_batch(
    *,
    engine: AsyncLLMEngine,
    prompt_token_ids: Sequence[list[int]],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None,
    priority: int,
) -> list[RequestOutput]:
    tasks = [
        asyncio.create_task(
            _generate_once(
                engine=engine,
                prompt_token_ids=token_ids,
                sampling_params=sampling_params,
                lora_request=lora_request,
                priority=priority,
            )
        )
        for token_ids in prompt_token_ids
    ]
    return await asyncio.gather(*tasks)


def _format_engine_results(
    outputs: Sequence[RequestOutput | None],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for _ in outputs:
        results.append({"token_ids": [], "token_logprobs": [], "finish_reason": []})

    for idx, request_output in enumerate(outputs):
        if request_output is None:
            continue

        completions: list[CompletionOutput] = sorted(
            request_output.outputs, key=lambda c: c.index
        )
        for completion in completions:
            logprob_entries: Sequence[dict[int, Logprob]] | None = completion.logprobs
            if logprob_entries is None:
                raise ValueError(
                    "Completion is missing logprob data. Ensure logprobs > 0."
                )
            token_logprobs: list[float] = []
            for token_id, token_logprob in zip(completion.token_ids, logprob_entries):
                chosen = token_logprob.get(token_id)
                if chosen is None:
                    if not token_logprob:
                        raise ValueError("No logprob candidates available")
                    chosen = min(token_logprob.values(), key=lambda lp: lp.logprob)
                token_logprobs.append(float(chosen.logprob))

            results[idx]["token_ids"].append(list(completion.token_ids))
            results[idx]["token_logprobs"].append(token_logprobs)
            results[idx]["finish_reason"].append(completion.finish_reason or "length")

    return results


def _sampling_params_with_n(request_params: dict[str, Any]) -> tuple[SamplingParams, int]:
    params = dict(request_params or {})
    requested = int(params.get("n", 1) or 1)
    params["n"] = 1
    return SamplingParams(**params), max(requested, 1)


def _expand_prompts(prompts: Sequence[list[int]], n: int) -> list[list[int]]:
    if n <= 1:
        return list(prompts)
    expanded: list[list[int]] = []
    for prompt in prompts:
        expanded.extend([prompt] * n)
    return expanded


def _regroup_results(
    expanded_results: list[dict[str, Any]], n: int, original_count: int
) -> list[dict[str, Any]]:
    if n <= 1:
        return expanded_results
    grouped = [
        {"token_ids": [], "token_logprobs": [], "finish_reason": []}
        for _ in range(original_count)
    ]
    for idx, result in enumerate(expanded_results):
        target = grouped[idx // n]
        target["token_ids"].extend(result.get("token_ids", []))
        target["token_logprobs"].extend(result.get("token_logprobs", []))
        target["finish_reason"].extend(result.get("finish_reason", []))
    return grouped


class EngineState:
    def __init__(self, *, engine: AsyncLLMEngine) -> None:
        self.engine: AsyncLLMEngine = engine
        self.lora_registry: dict[str, _RegisteredLoRA] = {}
        self._lora_int_counter: int = 0


def create_app(*, engine_args: AsyncEngineArgs) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: create engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        app.state.engine_state = EngineState(engine=engine)
        try:
            yield
        finally:
            # Shutdown
            app.state.engine_state.engine.shutdown_background_loop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/lora/register")
    async def register_lora(request: LoraManageRequest) -> dict[str, Any]:
        state: EngineState = app.state.engine_state
        path = request.lora_path
        if not path:
            raise HTTPException(status_code=400, detail="lora_path is required")
        cached = state.lora_registry.get(path)
        if cached is not None:
            return {"ok": True}
        adapter_name = f"lora_{uuid.uuid4().hex[:8]}"
        signature = _checkpoint_signature(path)
        state._lora_int_counter += 1
        req = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=state._lora_int_counter,
            lora_path=path,
        )
        await state.engine.add_lora(req)
        state.lora_registry[path] = _RegisteredLoRA(
            request=req,
            checkpoint_path=path,
            signature=signature,
        )
        return {"ok": True}

    @app.post("/lora/unregister")
    async def unregister_lora(request: LoraManageRequest) -> dict[str, Any]:
        state: EngineState = app.state.engine_state
        path = request.lora_path
        if not path:
            raise HTTPException(status_code=400, detail="lora_path is required")
        cached = state.lora_registry.get(path)
        if cached is None:
            return {"ok": True, "removed": []}
        await state.engine.remove_lora(cached.request.lora_int_id)
        state.lora_registry.pop(path, None)
        return {"ok": True, "removed": [path]}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        state: EngineState = app.state.engine_state
        if not request.prompt_token_ids:
            return GenerateResponse(ok=True, results=[])

        sampling_params, requested_n = _sampling_params_with_n(request.sampling_params)
        lora_request: LoRARequest | None = None
        if request.lora_path:
            cached = state.lora_registry.get(request.lora_path)
            if cached is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown LoRA path '{request.lora_path}'.",
                )
            lora_request = cached.request

        try:
            expanded_prompts = _expand_prompts(request.prompt_token_ids, requested_n)
            outputs = await _generate_batch(
                engine=state.engine,
                prompt_token_ids=expanded_prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
                priority=int(request.priority or 0),
            )
            expanded_results = _format_engine_results(outputs)
            results = _regroup_results(
                expanded_results, requested_n, len(request.prompt_token_ids)
            )
            return GenerateResponse(ok=True, results=results)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/generate_one", response_model=GenerateSingleResponse)
    async def generate_one(request: GenerateSingleRequest) -> GenerateSingleResponse:
        state: EngineState = app.state.engine_state
        if not request.prompt_token_ids:
            empty = CompletionResult(token_ids=[], token_logprobs=[], finish_reason=[])
            return GenerateSingleResponse(ok=True, result=empty)

        sampling_params, requested_n = _sampling_params_with_n(request.sampling_params)
        lora_request: LoRARequest | None = None
        if request.lora_path:
            cached = state.lora_registry.get(request.lora_path)
            if cached is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown LoRA path '{request.lora_path}'.",
                )
            lora_request = cached.request

        try:
            expanded_prompts = _expand_prompts([request.prompt_token_ids], requested_n)
            outputs = await _generate_batch(
                engine=state.engine,
                prompt_token_ids=expanded_prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
                priority=int(request.priority or 0),
            )
            expanded_results = _format_engine_results(outputs)
            results = _regroup_results(expanded_results, requested_n, 1)
            result = CompletionResult(**results[0])
            return GenerateSingleResponse(ok=True, result=result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


class InProcessServer:
    """
    In-process async API that mirrors the FastAPI endpoints with JSON I/O.
    """

    def __init__(self, *, engine: AsyncLLMEngine) -> None:
        self._state = EngineState(engine=engine)

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> "InProcessServer":
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return cls(engine=engine)

    async def close(self) -> None:
        self._state.engine.shutdown_background_loop()

    async def health(self) -> dict[str, Any]:
        return {"ok": True}

    async def lora_register(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = LoraManageRequest(**payload)
        path = request.lora_path
        if not path:
            raise HTTPException(status_code=400, detail="lora_path is required")
        cached = self._state.lora_registry.get(path)
        if cached is not None:
            return {"ok": True}
        adapter_name = f"lora_{uuid.uuid4().hex[:8]}"
        signature = _checkpoint_signature(path)
        self._state._lora_int_counter += 1
        req = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=self._state._lora_int_counter,
            lora_path=path,
        )
        await self._state.engine.add_lora(req)
        self._state.lora_registry[path] = _RegisteredLoRA(
            request=req,
            checkpoint_path=path,
            signature=signature,
        )
        return {"ok": True}

    async def lora_unregister(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = LoraManageRequest(**payload)
        path = request.lora_path
        if not path:
            raise HTTPException(status_code=400, detail="lora_path is required")
        cached = self._state.lora_registry.get(path)
        if cached is None:
            return {"ok": True, "removed": []}
        await self._state.engine.remove_lora(cached.request.lora_int_id)
        self._state.lora_registry.pop(path, None)
        return {"ok": True, "removed": [path]}

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = GenerateRequest(**payload)
        if not request.prompt_token_ids:
            return GenerateResponse(ok=True, results=[]).model_dump()

        sampling_params, requested_n = _sampling_params_with_n(request.sampling_params)
        lora_request: LoRARequest | None = None
        if request.lora_path:
            cached = self._state.lora_registry.get(request.lora_path)
            if cached is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown LoRA path '{request.lora_path}'.",
                )
            lora_request = cached.request

        expanded_prompts = _expand_prompts(request.prompt_token_ids, requested_n)
        outputs = await _generate_batch(
            engine=self._state.engine,
            prompt_token_ids=expanded_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            priority=int(request.priority or 0),
        )
        expanded_results = _format_engine_results(outputs)
        results = _regroup_results(
            expanded_results, requested_n, len(request.prompt_token_ids)
        )
        return GenerateResponse(ok=True, results=results).model_dump()

    async def generate_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = GenerateSingleRequest(**payload)
        if not request.prompt_token_ids:
            empty = CompletionResult(token_ids=[], token_logprobs=[], finish_reason=[])
            return GenerateSingleResponse(ok=True, result=empty).model_dump()

        sampling_params, requested_n = _sampling_params_with_n(request.sampling_params)
        lora_request: LoRARequest | None = None
        if request.lora_path:
            cached = self._state.lora_registry.get(request.lora_path)
            if cached is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown LoRA path '{request.lora_path}'.",
                )
            lora_request = cached.request

        expanded_prompts = _expand_prompts([request.prompt_token_ids], requested_n)
        outputs = await _generate_batch(
            engine=self._state.engine,
            prompt_token_ids=expanded_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            priority=int(request.priority or 0),
        )
        expanded_results = _format_engine_results(outputs)
        results = _regroup_results(expanded_results, requested_n, 1)
        result = CompletionResult(**results[0])
        return GenerateSingleResponse(ok=True, result=result).model_dump()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastAPI server that hosts a vLLM AsyncLLMEngine."
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8009, help="Bind port")
    parser.add_argument("--tensor-parallel-devices", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument(
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use data parallelism (DP). When false, use tensor parallelism (TP).",
    )

    return parser.parse_args()



def _build_engine_args(args: argparse.Namespace) -> AsyncEngineArgs:
    """Convert config knobs into vLLM ``EngineArgs`` arguments."""
    import torch

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "enable_lora": True,
        "enforce_eager":True,
        "enable_prefix_caching":True
    }
    gpu_count = torch.cuda.device_count()
    if args.data_parallel:
        engine_kwargs["data_parallel_size"] = max(gpu_count, 1)
        engine_kwargs["tensor_parallel_size"] = 1
    else:
        engine_kwargs["tensor_parallel_size"] = max(gpu_count, 1)
    engine_kwargs["max_lora_rank"] = args.max_lora_rank

    if args.gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.swap_space is not None:
        engine_kwargs["swap_space"] = args.swap_space
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    return AsyncEngineArgs(**engine_kwargs)


def main() -> None:
    args = _parse_args()
    engine_args = _build_engine_args(args)
    app = create_app(engine_args=engine_args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
