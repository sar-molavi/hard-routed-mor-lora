# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LoRA-Mixer model for Llama-family backbones."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    default_weight_loader,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    maybe_remap_kv_scale_name,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.sequence import IntermediateTensors
try:
    from vllm.v1.attention.backend import AttentionType
except Exception:  # pragma: no cover - fallback for older vLLM layouts
    try:
        from vllm.attention import AttentionType  # type: ignore
    except Exception:
        class AttentionType(str, Enum):
            DECODER = "decoder"
            ENCODER = "encoder"
            ENCODER_ONLY = "encoder_only"
            ENCODER_DECODER = "encoder_decoder"

from vllm.model_executor.models.llama import LlamaAttention, LlamaForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)


@dataclass(frozen=True)
class LoRAMixerHFConfig:
    expert_paths: list[str]
    checkpoint_path: str | None
    base_model_name_or_path: str | None
    base_model_revision: str | None
    base_model_cache_dir: str | None
    attention_lora_path: str | None
    top_k: int
    normalize_router_weights: bool
    apply_hard: bool | None
    jitter_noise: float
    router_shared_across_layers: bool
    num_layers: int | None


def _resolve_path(path: str, base_dir: Path | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if base_dir is not None and base_dir.is_dir():
        return (base_dir / candidate).resolve()
    return candidate.resolve()


def _resolve_expert_file(path: str, base_dir: Path | None) -> Path:
    candidate = _resolve_path(path, base_dir)
    if candidate.is_dir():
        adapter_file = candidate / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(
                f"Expected adapter_model.safetensors in {candidate}"
            )
        return adapter_file
    return candidate


def _resolve_checkpoint_file(path: str, base_dir: Path | None) -> Path:
    candidate = _resolve_path(path, base_dir)
    if candidate.is_dir():
        checkpoint_file = candidate / "lora_mixer.pth"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Expected lora_mixer.pth in {candidate}")
        return checkpoint_file
    return candidate


def _find_lora_key(keys: list[str], layer_idx: int, proj_name: str, suffix: str) -> str:
    target = f"layers.{layer_idx}.mlp.{proj_name}.lora_{suffix}.weight"
    matches = [key for key in keys if key.endswith(target)]
    if not matches:
        raise KeyError(
            f"Missing LoRA {suffix} key for {proj_name} in layer {layer_idx}"
        )
    return min(matches, key=len)


def _find_attn_lora_key(
    keys: list[str], layer_idx: int, proj_name: str, suffix: str
) -> str | None:
    target = f"layers.{layer_idx}.self_attn.{proj_name}.lora_{suffix}.weight"
    matches = [key for key in keys if key.endswith(target)]
    if not matches:
        return None
    return min(matches, key=len)


def _load_lora_scalings_and_rank(expert_paths: list[str]) -> tuple[list[float], int]:
    scalings: list[float] = []
    ranks: list[int] = []
    for expert_path in expert_paths:
        expert_dir = Path(expert_path).expanduser()
        if expert_dir.is_file():
            expert_dir = expert_dir.parent
        config_path = expert_dir / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing PEFT adapter_config.json for expert: {expert_dir}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        r = payload.get("r")
        lora_alpha = payload.get("lora_alpha")
        if not r or not lora_alpha:
            raise ValueError(
                f"Invalid PEFT config in {config_path}: missing r or lora_alpha."
            )
        ranks.append(int(r))
        scalings.append(float(lora_alpha) / float(r))
    if len(set(ranks)) != 1:
        raise ValueError(f"Inconsistent LoRA ranks across experts: {ranks}")
    return scalings, ranks[0]


def _parse_lora_mixer_config(config: LlamaConfig) -> LoRAMixerHFConfig:
    mixer = getattr(config, "lora_mixer", None)
    if mixer is None:
        raise ValueError(
            "Missing `lora_mixer` configuration in the model config. "
            "Expected a dict with at least `expert_paths`."
        )
    if isinstance(mixer, str):
        mixer = {"checkpoint_path": mixer}
    if not isinstance(mixer, dict):
        raise TypeError("`lora_mixer` must be a dict or a checkpoint path string.")

    expert_paths = mixer.get("expert_paths")
    if not expert_paths:
        raise ValueError("`lora_mixer.expert_paths` must be provided.")

    return LoRAMixerHFConfig(
        expert_paths=list(expert_paths),
        checkpoint_path=mixer.get("checkpoint_path")
        or mixer.get("checkpoint_dir")
        or mixer.get("lora_mixer_path"),
        base_model_name_or_path=mixer.get("base_model_name_or_path"),
        base_model_revision=mixer.get("base_model_revision"),
        base_model_cache_dir=mixer.get("base_model_cache_dir"),
        attention_lora_path=mixer.get("attention_lora_path")
        or mixer.get("attn_lora_path")
        or mixer.get("lora_attn_path"),
        top_k=int(mixer.get("top_k", 1)),
        normalize_router_weights=bool(mixer.get("normalize_router_weights", False)),
        apply_hard=mixer.get("apply_hard"),
        jitter_noise=float(mixer.get("jitter_noise", 0.0) or 0.0),
        router_shared_across_layers=bool(
            mixer.get("router_shared_across_layers", False)
        ),
        num_layers=mixer.get("num_layers"),
    )


def _resolve_attention_lora_file(path: str, base_dir: Path | None) -> Path:
    candidate = _resolve_path(path, base_dir)
    if candidate.is_dir():
        adapter_file = candidate / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(
                f"Expected adapter_model.safetensors in {candidate}"
            )
        return adapter_file
    return candidate


def _extract_layer_idx_from_weight(name: str) -> int | None:
    parts = name.split(".")
    try:
        idx = parts.index("layers")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return None


def _collect_weight_files(folder: str) -> tuple[list[str], bool]:
    base = Path(folder)
    safetensors = sorted(str(p) for p in base.glob("*.safetensors"))
    if safetensors:
        safetensors = filter_files_not_needed_for_inference(safetensors)
        safetensors = filter_duplicate_safetensors_files(
            safetensors, str(base), SAFE_WEIGHTS_INDEX_NAME
        )
        return safetensors, True
    pt_files = sorted(
        str(p)
        for p in list(base.glob("*.bin")) + list(base.glob("*.pt"))
    )
    pt_files = filter_files_not_needed_for_inference(pt_files)
    return pt_files, False


def _iter_base_model_weights(
    *,
    model_name_or_path: str,
    cache_dir: str | None,
    revision: str | None,
):
    if os.path.isdir(model_name_or_path):
        folder = model_name_or_path
        weight_files, use_safetensors = _collect_weight_files(folder)
    else:
        folder = download_weights_from_hf(
            model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors"],
            revision=revision,
        )
        weight_files, use_safetensors = _collect_weight_files(folder)
        if not weight_files:
            folder = download_weights_from_hf(
                model_name_or_path,
                cache_dir=cache_dir,
                allow_patterns=["*.bin", "*.pt"],
                revision=revision,
            )
            weight_files, use_safetensors = _collect_weight_files(folder)

    if not weight_files:
        raise FileNotFoundError(
            f"No model weight files found for {model_name_or_path}."
        )

    if use_safetensors:
        return safetensors_weights_iterator(weight_files, use_tqdm_on_load=False)
    return pt_weights_iterator(weight_files, use_tqdm_on_load=False)


class LoRAMixerRouter(nn.Module):
    """Mixtral-style router: linear gate + top-k softmax."""

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        top_k: int,
        normalize: bool,
        jitter_noise: float = 0.0,
        apply_hard: bool | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._force_expert_idx = 2 if apply_hard else None
        self.normalize = normalize
        self.num_experts = num_experts
        self.top_k = top_k
        self.apply_hard = apply_hard
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(in_features, self.num_experts, bias=False)
        if params_dtype is not None:
            self.gate.to(dtype=params_dtype)

    def _soft_routing(
        self, scores: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        if self.normalize:
            topk_weights = F.normalize(topk_weights, p=1, dim=-1)
        return topk_weights

    def _hard_routing(
        self, scores: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        hard_weights = torch.zeros_like(scores)
        hard_weights.scatter_(dim=-1, index=topk_indices, value=1.0)
        router_weights = hard_weights - scores.detach() + scores
        topk_weights = router_weights.gather(dim=-1, index=topk_indices)
        return topk_weights

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        scores = F.softmax(self.gate(hidden_states), dim=-1)
        if self._force_expert_idx is not None:
            if not 0 <= self._force_expert_idx < self.num_experts:
                raise ValueError(
                    f"force_expert_idx {self._force_expert_idx} is out of range "
                    f"for num_experts={self.num_experts}."
                )
            topk_indices = torch.full(
                (scores.shape[0], self.top_k),
                self._force_expert_idx,
                device=scores.device,
                dtype=torch.long,
            )
            topk_weights = torch.ones(
                (scores.shape[0], self.top_k),
                device=scores.device,
                dtype=scores.dtype,
            )
        else:
            topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        if self.apply_hard:
            topk_weights = self._hard_routing(scores, topk_indices, topk_weights)
        else:
            topk_weights = self._soft_routing(scores, topk_indices, topk_weights)
        return topk_weights, topk_indices


class LoRAExperts(nn.Module):
    """Apply routed LoRA experts for a single projection."""

    def __init__(
        self,
        *,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        lora_scaling: torch.Tensor,
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(lora_A)
        self.lora_B = nn.Parameter(lora_B)
        self.register_buffer("lora_scaling", lora_scaling.to(lora_A))
        self.num_experts = lora_A.shape[0]

    def load_lora_weights(
        self,
        *,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        lora_scaling: torch.Tensor,
    ) -> None:
        if self.lora_A.shape != lora_A.shape or self.lora_B.shape != lora_B.shape:
            raise ValueError("Loaded LoRA expert shapes do not match initialized ones.")
        self.lora_A.data.copy_(lora_A)
        self.lora_B.data.copy_(lora_B)
        self.lora_scaling.data.copy_(lora_scaling.to(self.lora_scaling))

    def forward(
        self,
        x: torch.Tensor,
        *,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        token_count, hidden_dim = x.shape
        k = topk_indices.shape[-1]
        output_dim = self.lora_B.shape[1]

        flat_topk_indices = topk_indices.reshape(token_count, k)
        flat_topk_weights = topk_weights.reshape(token_count, k)

        final = x.new_zeros((token_count, output_dim))

        for expert_idx in range(self.num_experts):
            mask = flat_topk_indices == expert_idx
            if not mask.any():
                continue
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            current_state = x[token_idx]
            lora_A = self.lora_A[expert_idx]
            lora_B = self.lora_B[expert_idx]
            delta = F.linear(current_state, lora_A)
            delta = F.linear(delta, lora_B)
            delta = delta * self.lora_scaling[expert_idx]
            weights = flat_topk_weights[token_idx, k_idx].unsqueeze(-1)
            final.index_add_(0, token_idx, delta * weights)

        return final


class LoRAMixerMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None,
        bias: bool,
        prefix: str,
        mixer_state: "LoRAMixerState",
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

        self._router = None
        self._lora_experts: dict[str, LoRAExperts] | None = None
        if layer_idx < mixer_state.num_mixer_layers:
            if mixer_state.router_shared_across_layers:
                if mixer_state.shared_router is None:
                    raise ValueError("Shared router is not initialized.")
                router = mixer_state.shared_router
            else:
                router = LoRAMixerRouter(
                    hidden_size,
                    mixer_state.num_experts,
                    mixer_state.top_k,
                    mixer_state.normalize_router_weights,
                    jitter_noise=mixer_state.jitter_noise,
                    apply_hard=mixer_state.apply_hard,
                    params_dtype=mixer_state.params_dtype,
                )
                mixer_state.router_layers[f"layer{layer_idx}"] = router
            self._router = router

            lora_scaling = torch.tensor(
                mixer_state.lora_scalings, dtype=mixer_state.params_dtype
            )
            lora_experts: dict[str, LoRAExperts] = {}
            for proj_name, in_dim, out_dim in (
                ("gate_proj", hidden_size, intermediate_size),
                ("up_proj", hidden_size, intermediate_size),
                ("down_proj", intermediate_size, hidden_size),
            ):
                key = f"layer{layer_idx}_{proj_name}"
                lora_A = torch.zeros(
                    (mixer_state.num_experts, mixer_state.lora_rank, in_dim),
                    dtype=mixer_state.params_dtype,
                )
                lora_B = torch.zeros(
                    (mixer_state.num_experts, out_dim, mixer_state.lora_rank),
                    dtype=mixer_state.params_dtype,
                )
                expert = LoRAExperts(
                    lora_A=lora_A,
                    lora_B=lora_B,
                    lora_scaling=lora_scaling,
                )
                mixer_state.moe_layers[key] = expert
                lora_experts[proj_name] = expert
            self._lora_experts = lora_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if self._router is None or self._lora_experts is None:
            gate_up = self.act_fn(gate_up)
            down, _ = self.down_proj(gate_up)
            return down

        topk_weights, topk_indices = self._router(x)
        gate_delta = self._lora_experts["gate_proj"](
            x, topk_weights=topk_weights, topk_indices=topk_indices
        )
        up_delta = self._lora_experts["up_proj"](
            x, topk_weights=topk_weights, topk_indices=topk_indices
        )
        gate_up[:, : self.intermediate_size] += gate_delta
        gate_up[:, self.intermediate_size :] += up_delta

        hidden = self.act_fn(gate_up)
        down, _ = self.down_proj(hidden)
        down = down + self._lora_experts["down_proj"](
            hidden, topk_weights=topk_weights, topk_indices=topk_indices
        )
        return down


@dataclass
class LoRAMixerState:
    moe_layers: nn.ModuleDict
    router_layers: nn.ModuleDict
    shared_router: LoRAMixerRouter | None
    num_experts: int
    top_k: int
    normalize_router_weights: bool
    apply_hard: bool | None
    jitter_noise: float
    router_shared_across_layers: bool
    num_mixer_layers: int
    lora_scalings: list[float]
    lora_rank: int
    params_dtype: torch.dtype


class LoRAMixerDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        mixer_state: LoRAMixerState,
        config: LlamaConfig | None = None,
        attn_layer_type: type[nn.Module] = LlamaAttention,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = attn_layer_type(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )

        layer_idx = extract_layer_index(prefix)
        self.mlp = LoRAMixerMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
            mixer_state=mixer_state,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LoRAMixerModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        mixer_config = _parse_lora_mixer_config(config)
        self.mixer_config = mixer_config

        params_dtype = vllm_config.model_config.dtype
        base_dir = None
        if mixer_config.base_model_name_or_path and os.path.isdir(
            mixer_config.base_model_name_or_path
        ):
            base_dir = Path(mixer_config.base_model_name_or_path)
        else:
            name_or_path = getattr(config, "_name_or_path", None)
            if name_or_path:
                base_dir = Path(name_or_path)

        expert_paths = [
            str(_resolve_expert_file(path, base_dir))
            for path in mixer_config.expert_paths
        ]
        lora_scalings, lora_rank = _load_lora_scalings_and_rank(expert_paths)

        num_mixer_layers = (
            config.num_hidden_layers
            if mixer_config.num_layers is None
            else int(mixer_config.num_layers)
        )
        if num_mixer_layers < 0 or num_mixer_layers > config.num_hidden_layers:
            raise ValueError(
                "lora_mixer.num_layers must be between 0 and "
                f"{config.num_hidden_layers}."
            )

        self.moe_layers = nn.ModuleDict()
        self.router_layers = nn.ModuleDict()
        if mixer_config.router_shared_across_layers:
            self.shared_router = LoRAMixerRouter(
                config.hidden_size,
                len(expert_paths),
                mixer_config.top_k,
                mixer_config.normalize_router_weights,
                jitter_noise=mixer_config.jitter_noise,
                apply_hard=mixer_config.apply_hard,
                params_dtype=params_dtype,
            )
        else:
            self.shared_router = None
        self.router_shared_across_layers = mixer_config.router_shared_across_layers
        self.expert_paths = expert_paths
        self.lora_scalings = lora_scalings
        self.attn_lora_state: dict[str, torch.Tensor] | None = None
        self.attn_lora_scaling: float | None = None
        self.attn_lora_rank: int | None = None
        self.attn_lora_path: Path | None = None

        attn_lora_candidate = mixer_config.attention_lora_path
        if attn_lora_candidate is None and mixer_config.checkpoint_path:
            checkpoint_path = _resolve_checkpoint_file(
                mixer_config.checkpoint_path, base_dir
            )
            checkpoint_dir = (
                checkpoint_path
                if checkpoint_path.is_dir()
                else checkpoint_path.parent
            )
            if (checkpoint_dir / "adapter_model.safetensors").is_file():
                attn_lora_candidate = str(checkpoint_dir)

        if attn_lora_candidate:
            self.attn_lora_path = _resolve_attention_lora_file(
                attn_lora_candidate, base_dir
            )
            self.attn_lora_state = load_file(str(self.attn_lora_path), device="cpu")
            attn_scalings, attn_rank = _load_lora_scalings_and_rank(
                [str(self.attn_lora_path)]
            )
            self.attn_lora_scaling = attn_scalings[0]
            self.attn_lora_rank = attn_rank

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        mixer_state = LoRAMixerState(
            moe_layers=self.moe_layers,
            router_layers=self.router_layers,
            shared_router=self.shared_router,
            num_experts=len(expert_paths),
            top_k=mixer_config.top_k,
            normalize_router_weights=mixer_config.normalize_router_weights,
            apply_hard=mixer_config.apply_hard,
            jitter_noise=mixer_config.jitter_noise,
            router_shared_across_layers=mixer_config.router_shared_across_layers,
            num_mixer_layers=num_mixer_layers,
            lora_scalings=lora_scalings,
            lora_rank=lora_rank,
            params_dtype=params_dtype,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LoRAMixerDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                mixer_state=mixer_state,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def apply_attention_lora(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if self.attn_lora_state is None or self.attn_lora_scaling is None:
            return weights

        attn_keys = list(self.attn_lora_state.keys())
        scaling = float(self.attn_lora_scaling)

        def _generator():
            for name, weight in weights:
                if not name.endswith(
                    (
                        ".self_attn.q_proj.weight",
                        ".self_attn.k_proj.weight",
                        ".self_attn.v_proj.weight",
                        ".self_attn.o_proj.weight",
                    )
                ):
                    yield name, weight
                    continue

                layer_idx = _extract_layer_idx_from_weight(name)
                if layer_idx is None:
                    yield name, weight
                    continue

                proj_name = name.split(".")[-2]
                key_A = _find_attn_lora_key(attn_keys, layer_idx, proj_name, "A")
                key_B = _find_attn_lora_key(attn_keys, layer_idx, proj_name, "B")
                if key_A is None or key_B is None:
                    yield name, weight
                    continue

                lora_A = self.attn_lora_state[key_A].to(weight)
                lora_B = self.attn_lora_state[key_B].to(weight)
                delta = torch.matmul(lora_B, lora_A) * scaling
                weight = weight + delta
                yield name, weight

        return _generator()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(
                positions, hidden_states, residual, **extra_layer_kwargs
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def load_mixer_weights(self) -> None:
        base_dir = None
        name_or_path = getattr(self.config, "_name_or_path", None)
        if name_or_path:
            base_dir = Path(name_or_path)

        experts = []
        expert_keys = []
        for path in self.expert_paths:
            file_path = _resolve_expert_file(path, base_dir)
            expert = load_file(str(file_path), device="cpu")
            experts.append(expert)
            expert_keys.append(list(expert.keys()))

        max_layers = (
            self.mixer_config.num_layers
            if self.mixer_config.num_layers is not None
            else self.config.num_hidden_layers
        )
        for layer_idx in range(min(int(max_layers), self.config.num_hidden_layers)):
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                key = f"layer{layer_idx}_{proj_name}"
                if key not in self.moe_layers:
                    continue
                moe = self.moe_layers[key]
                lora_A_tensors = []
                lora_B_tensors = []
                for expert, keys in zip(experts, expert_keys, strict=True):
                    key_A = _find_lora_key(keys, layer_idx, proj_name, "A")
                    key_B = _find_lora_key(keys, layer_idx, proj_name, "B")
                    lora_A_tensors.append(expert[key_A].to(moe.lora_A))
                    lora_B_tensors.append(expert[key_B].to(moe.lora_B))

                lora_A = torch.stack(lora_A_tensors, dim=0)
                lora_B = torch.stack(lora_B_tensors, dim=0)
                lora_scaling = torch.tensor(
                    self.lora_scalings,
                    device=moe.lora_A.device,
                    dtype=moe.lora_A.dtype,
                )
                moe.load_lora_weights(
                    lora_A=lora_A, lora_B=lora_B, lora_scaling=lora_scaling
                )

        if not self.mixer_config.checkpoint_path:
            return

        checkpoint_path = _resolve_checkpoint_file(
            self.mixer_config.checkpoint_path, base_dir
        )
        payload = torch.load(checkpoint_path, map_location="cpu")

        if "routers" in payload:
            router_payload = payload["routers"]
            router_shared = payload.get("router_shared")
            if router_shared is None and isinstance(router_payload, dict):
                if "shared" in router_payload and not any(
                    key.startswith("layer") for key in router_payload.keys()
                ):
                    router_shared = True
            if router_shared:
                if self.shared_router is None:
                    raise ValueError("Shared router is not initialized.")
                shared_state = router_payload.get("shared", router_payload)
                self.shared_router.load_state_dict(shared_state, strict=False)
            else:
                self.router_layers.load_state_dict(router_payload, strict=False)

        if "lora_params" in payload:
            self.moe_layers.load_state_dict(payload["lora_params"], strict=False)


class LoRAMixerForCausalLM(LlamaForCausalLM):
    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LoRAMixerDecoderLayer,
    ):
        return LoRAMixerModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model.mixer_config.base_model_name_or_path:
            weights = _iter_base_model_weights(
                model_name_or_path=self.model.mixer_config.base_model_name_or_path,
                cache_dir=self.model.mixer_config.base_model_cache_dir,
                revision=self.model.mixer_config.base_model_revision,
            )
        weights = self.model.apply_attention_lora(weights)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                (["lm_head."] if self.config.tie_word_embeddings else [])
                + [
                    "model.moe_layers.",
                    "model.router_layers.",
                    "model.shared_router.",
                ]
            ),
        )
        loaded = loader.load_weights(weights)
        self.model.load_mixer_weights()
        if loaded is None:
            return loaded

        # Mark mixer params as loaded so vLLM strict check does not fail.
        mixer_prefixes = (
            "model.moe_layers.",
            "model.router_layers.",
            "model.shared_router.",
        )
        loaded.update(
            name
            for name, _ in self.named_parameters()
            if name.startswith(mixer_prefixes)
        )
        return loaded
