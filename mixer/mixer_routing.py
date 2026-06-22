"""LoRA-Mixer variant that patches FFN (MLP) projections for Llama models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

from .moe_routing import RSL, LoRAExperts


def _make_key(layer_idx, proj_name):
    return f"layer{layer_idx}_{proj_name}"


def _enable_gradient_checkpointing(model) -> None:
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()


def _enable_lora_attention(*, model, lora_kwargs):
    if "target_modules" not in lora_kwargs:
        lora_kwargs = dict(lora_kwargs)
        lora_kwargs["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, peft_config)
    return model


def _resolve_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model"):
            return base.model
        return base
    if hasattr(model, "model"):
        return model.model
    return model


def _get_layers(model: nn.Module):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "layers"):
            return base.layers
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers
    raise AttributeError("Could not locate model layers for FFN patching.")


def _get_num_layers(model):
    return len(_get_layers(model))


def _find_lora_key(keys: list[str], layer_idx: int, proj_name: str, suffix: str) -> str:
    target = f"layers.{layer_idx}.mlp.{proj_name}.lora_{suffix}.weight"
    matches = [k for k in keys if k.endswith(target)]
    if not matches:
        raise KeyError(
            f"Missing LoRA {suffix} key for {proj_name} in layer {layer_idx}"
        )
    return min(matches, key=len)


def _checkpoint(function, *args):
    try:
        return checkpoint.checkpoint(function, *args, use_reentrant=False)
    except TypeError:
        return checkpoint.checkpoint(function, *args)


def _load_lora_scalings(expert_paths: list[str]) -> list[float]:
    scalings: list[float] = []
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
        scalings.append(float(lora_alpha) / float(r))
    return scalings


def token_level_consistency_loss(
    *, choices: list[torch.Tensor], attention_mask: torch.Tensor, gamma: float = 1
):
    """
    Computes the Gini score as consistency loss at token level.
    """
    # list[(batch_size, seq_len, n_experts)] -> (batch_size, seq_len, n_layers, n_experts)
    choices = torch.stack(choices, dim=2)

    # Mean over layers
    # (batch_size, seq_len, n_layers, n_experts) -> (batch_size, seq_len, n_experts)
    probs = nn.functional.normalize(choices.mean(dim=2), p=1, dim=-1)

    # Token-level Gini
    # (batch_size, seq_len, n_experts) -> (batch_size, seq_len)
    gini = 1 - probs.pow(2).sum(dim=-1)

    if attention_mask is None:
        loss = gini.mean()
    else:
        mask = attention_mask.to(dtype=gini.dtype)

        # Proper masked mean over tokens AND batch
        loss = (gini * mask).sum() / mask.sum().clamp(min=1.0)

    return loss * gamma


def sequence_level_consistency_loss(
    choices: list[torch.Tensor], attention_mask: torch.Tensor, gamma: float = 1
):
    """
    Computes the Gini score as consistency loss at sequence level.

    Sequence-level means: build one expert-distribution per sequence by
    averaging routing choices across tokens AND layers, then compute Gini.
    """
    # list[(batch_size, seq_len, n_experts)] -> (batch_size, seq_len, n_layers, n_experts)
    choices = torch.stack(choices, dim=2)

    if attention_mask is None:
        # Mean across tokens and layers:
        # (batch_size, seq_len, n_layers, n_experts) -> (batch_size, n_experts)
        probs = nn.functional.normalize(choices.mean(dim=(1, 2)), p=1, dim=-1)

        # (batch_size, n_experts) -> (batch_size,)
        gini = 1 - probs.pow(2).sum(dim=-1)

        loss = gini.mean() * gamma
        return loss

    # Masked case: average across tokens (masked) and layers (unmasked)

    # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1, 1)
    attention_mask = attention_mask.to(dtype=choices.dtype)
    mask = attention_mask.unsqueeze(-1).unsqueeze(-1)

    # Zero out padded tokens
    masked_choices = choices * mask

    # Sum over tokens and layers:
    # (batch_size, seq_len, n_layers, n_experts) -> (batch_size, n_experts)
    summed = masked_choices.sum(dim=(1, 2))

    # Denominator: number of *valid tokens* times number of layers
    # valid_tokens: (batch_size, 1)
    valid_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    denom = valid_tokens * choices.size(2)  # n_layers

    mean_choices = summed / denom  # (batch_size, n_experts)

    probs = nn.functional.normalize(mean_choices, p=1, dim=-1)

    gini = 1 - probs.pow(2).sum(dim=-1)  # (batch_size,)

    loss = gini.mean() * gamma
    return loss


class LoRAFFNBlock:
    """Apply per-layer routing and LoRA deltas across the full FFN block."""

    def __init__(
        self,
        *,
        router: RSL,
        mlp: nn.Module,
        lora_experts: dict[str, LoRAExperts],
        enable_gradient_checkpointing: bool,
    ) -> None:
        self.router = router
        self.mlp = mlp
        self.lora_experts = lora_experts
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.act_fn = getattr(mlp, "act_fn", None) or getattr(
            mlp, "activation_fn", None
        )
        if self.act_fn is None:
            raise AttributeError("MLP module must expose an activation function.")

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        choices: list[torch.Tensor] | None,
        routing_weights: list[torch.Tensor] | None,
        routing_detach: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shape summary:
            x: (batch, seq, in_features)
            attention_mask: (batch, seq) or None
            output: (batch, seq, out_features), scalar aux_loss
        """

        def full_forward(input_x: torch.Tensor):
            scores, aux_loss, topk_indices, topk_weights, router_weights = self.router(
                input_x, attention_mask=attention_mask
            )
            if aux_loss is None:
                aux_loss = input_x.new_zeros(())

            # (batch, seq, in_features) -> (batch, seq, out_features)
            gate = self.mlp.gate_proj(input_x)
            # (batch, seq, in_features) -> (batch, seq, out_features)
            gate = gate + self.lora_experts["gate_proj"](
                input_x, topk_weights=topk_weights, topk_indices=topk_indices
            )
            # (batch, seq, in_features) -> (batch, seq, out_features)
            up = self.mlp.up_proj(input_x)
            # (batch, seq, in_features) -> (batch, seq, out_features)
            up = up + self.lora_experts["up_proj"](
                input_x, topk_weights=topk_weights, topk_indices=topk_indices
            )
            # (batch, seq, out_features) -> (batch, seq, out_features)
            hidden = self.act_fn(gate) * up
            # (batch, seq, out_features) -> (batch, seq, in_features)
            down = self.mlp.down_proj(hidden)
            # (batch, seq, out_features) -> (batch, seq, in_features)
            down = down + self.lora_experts["down_proj"](
                hidden, topk_weights=topk_weights, topk_indices=topk_indices
            )
            return down, aux_loss, scores, router_weights

        if (
            self.enable_gradient_checkpointing
            and self.router.training
            and x.requires_grad
        ):
            down, aux_loss, scores, router_weights = _checkpoint(full_forward, x)
        else:
            down, aux_loss, scores, router_weights = full_forward(x)

        if choices is not None:
            choices.append(scores)
        if routing_weights is not None:
            if routing_detach:
                router_weights = router_weights.detach()
            routing_weights.append(router_weights)

        return down, aux_loss


class LoRAMixerFFN(nn.Module):
    """
    Wraps a frozen base transformer, augmenting its FFN (MLP) projections with
    dynamically routed LoRA experts.
    """

    def __init__(
        self,
        base_model: nn.Module,
        expert_paths: list[str],
        num_layers: int,
        top_k: int,
        freeze_router: bool,
        freeze_experts: bool,
        enable_lora_attn: bool,
        normalize_router_weights: bool,
        proj_names: list[str] | None = None,
        alpha: float = 0.0,
        token_gamma: float = 0.0,
        sequence_gamma: float = 0.0,
        lora_kwargs: dict[str, Any] | None = None,
        enable_gradient_checkpointing: bool = False,
        jitter_noise: float | None = None,
        apply_hard: bool | None = None,
        router_shared_across_layers: bool = False,
    ):
        super().__init__()
        num_layers = _get_num_layers(base_model) if num_layers is None else num_layers
        self.normalize_router_weights = normalize_router_weights
        self.enable_lora_attn = enable_lora_attn
        self.jitter_noise = jitter_noise
        self.apply_hard = apply_hard
        self.router_shared_across_layers = router_shared_across_layers

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.top_k = top_k
        self.proj_names = proj_names or ["gate_proj", "up_proj", "down_proj"]
        self.enable_gradient_checkpointing = enable_gradient_checkpointing

        self.num_experts = len(expert_paths)

        self._freeze_model(base_model)

        if enable_lora_attn:
            if lora_kwargs is None:
                raise ValueError(
                    "lora_kwargs must be provided when enable_lora_attn=True."
                )
            base_model = _enable_lora_attention(
                model=base_model, lora_kwargs=lora_kwargs
            )

        if enable_gradient_checkpointing:
            _enable_gradient_checkpointing(base_model)

        self.wrapped_base_model = _resolve_base_model(base_model)
        self.wrapped_model = base_model

        self.num_layers = num_layers
        self.aux_loss = 0.0
        self._attention_mask: torch.Tensor | None = None
        self._choices: list[torch.Tensor] | None = None
        self._routing_weights: list[torch.Tensor] | None = None
        self.routing_distributions: torch.Tensor | None = None
        self._routing_detach: bool = True
        self.alpha = alpha
        self.token_gamma = token_gamma
        self.sequence_gamma = sequence_gamma
        self.freeze_router = freeze_router
        self.freeze_experts = freeze_experts

        self.moe_layers = nn.ModuleDict()
        self.router_layers = nn.ModuleDict()
        self.shared_router: RSL | None = None
        self.ffn_blocks: dict[int, LoRAFFNBlock] = {}

        self.expert_usage_list: list[dict[str, float]] = []

        self._init_routers_and_lora(expert_paths)

        if freeze_router:
            self._freeze_router()
        else:
            self._unfreeze_router()

        if freeze_experts:
            self._freeze_experts()
        else:
            self._unfreeze_experts()

        self._monkey_patch_linear_layers()

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_experts(self):
        for moe in self.moe_layers.values():
            for param in moe.lora_parameters():
                param.requires_grad = True
        self.freeze_experts = False

    def _freeze_experts(self):
        for moe in self.moe_layers.values():
            for param in moe.lora_parameters():
                param.requires_grad = False
        self.freeze_experts = True

    def _unfreeze_router(self):
        if self.router_shared_across_layers:
            if self.shared_router is not None:
                for param in self.shared_router.parameters():
                    param.requires_grad = True
        else:
            for router in self.router_layers.values():
                for param in router.parameters():
                    param.requires_grad = True
        self.freeze_router = False

    def _freeze_router(self):
        if self.router_shared_across_layers:
            if self.shared_router is not None:
                for param in self.shared_router.parameters():
                    param.requires_grad = False
        else:
            for router in self.router_layers.values():
                for param in router.parameters():
                    param.requires_grad = False
        self.freeze_router = True

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Delegate gradient checkpointing to the wrapped base model.

        Hugging Face's Trainer checks for this method when gradient checkpointing
        is requested, so surface it here and forward the call.
        """
        if not hasattr(self.wrapped_model, "gradient_checkpointing_enable"):
            return

        kwargs = dict(gradient_checkpointing_kwargs or {})
        kwargs.setdefault("use_reentrant", False)

        self.wrapped_base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        """Mirror the HF API by delegating the disable call as well."""
        if hasattr(self.wrapped_base_model, "gradient_checkpointing_disable"):
            self.wrapped_base_model.gradient_checkpointing_disable()

    def _init_routers_and_lora(self, expert_paths: list[str]):
        if not expert_paths:
            raise ValueError("At least one expert checkpoint is required.")

        self.logger.info("\nInitializing router network and LoRA parameters...")
        experts = [load_file(path, device="cpu") for path in expert_paths]
        expert_keys = [list(expert.keys()) for expert in experts]
        lora_scalings = _load_lora_scalings(expert_paths)
        layers = _get_layers(self.wrapped_base_model)
        shared_in_features: int | None = None

        for layer_idx in range(self.num_layers):
            layer_router: RSL | None = None
            for proj_name in self.proj_names:
                key = _make_key(layer_idx=layer_idx, proj_name=proj_name)
                try:
                    mlp = layers[layer_idx].mlp
                    proj = getattr(mlp, proj_name)
                    if not isinstance(proj, nn.Linear):
                        raise TypeError(
                            f"Expected nn.Linear, but {proj_name} is {type(proj)}"
                        )

                    param_device = proj.weight.device
                    param_dtype = proj.weight.dtype

                    if layer_router is None:
                        if self.router_shared_across_layers:
                            if self.shared_router is None:
                                shared_in_features = proj.in_features
                                self.shared_router = RSL(
                                    in_features=proj.in_features,
                                    num_experts=self.num_experts,
                                    top_k=self.top_k,
                                    normalize=self.normalize_router_weights,
                                    alpha=self.alpha,
                                    jitter_noise=self.jitter_noise,
                                    enable_gradient_checkpointing=self.enable_gradient_checkpointing,
                                    apply_hard=self.apply_hard,
                                ).to(param_device, param_dtype)
                            elif shared_in_features != proj.in_features:
                                raise ValueError(
                                    "Shared router requires consistent FFN in_features across layers."
                                )
                            layer_router = self.shared_router
                        else:
                            layer_router = RSL(
                                in_features=proj.in_features,
                                num_experts=self.num_experts,
                                top_k=self.top_k,
                                normalize=self.normalize_router_weights,
                                alpha=self.alpha,
                                jitter_noise=self.jitter_noise,
                                enable_gradient_checkpointing=self.enable_gradient_checkpointing,
                                apply_hard=self.apply_hard,
                            ).to(param_device, param_dtype)
                            self.router_layers[f"layer{layer_idx}"] = layer_router

                    lora_A_tensors: list[torch.Tensor] = []
                    lora_B_tensors: list[torch.Tensor] = []

                    for expert, keys in zip(experts, expert_keys, strict=True):
                        key_A = _find_lora_key(keys, layer_idx, proj_name, "A")
                        key_B = _find_lora_key(keys, layer_idx, proj_name, "B")
                        lora_A_tensors.append(
                            expert[key_A].to(device=param_device, dtype=param_dtype)
                        )
                        lora_B_tensors.append(
                            expert[key_B].to(device=param_device, dtype=param_dtype)
                        )

                    moe = LoRAExperts(
                        lora_A=torch.stack(lora_A_tensors, dim=0),
                        lora_B=torch.stack(lora_B_tensors, dim=0),
                        lora_scaling=torch.tensor(
                            lora_scalings, device=param_device, dtype=param_dtype
                        ),
                        enable_gradient_checkpointing=self.enable_gradient_checkpointing,
                    )
                    self.moe_layers[key] = moe
                except (AttributeError, KeyError) as err:
                    self.logger.warning(
                        "No layer %s proj %s due to error: %s",
                        layer_idx,
                        proj_name,
                        err,
                    )
                    raise err

    def _get_router(self, layer_idx: int) -> RSL:
        if self.router_shared_across_layers:
            if self.shared_router is None:
                raise ValueError("Shared router is not initialized.")
            return self.shared_router
        router_key = f"layer{layer_idx}"
        if router_key not in self.router_layers:
            raise KeyError(f"Missing router for layer {layer_idx}.")
        return self.router_layers[router_key]

    def _monkey_patch_linear_layers(self):
        """Replace the forward method of each FFN block to include LoRA mixing."""
        self.logger.info("\nReplacing forward methods of FFN layers...")
        layers = _get_layers(self.wrapped_base_model)
        for layer_idx in range(self.num_layers):
            mlp = layers[layer_idx].mlp
            lora_experts: dict[str, LoRAExperts] = {}
            for proj_name in self.proj_names:
                key = _make_key(layer_idx=layer_idx, proj_name=proj_name)
                if key not in self.moe_layers:
                    raise Exception(f"{key} is not in self.moe_layers.")
                lora_experts[proj_name] = self.moe_layers[key]

            block = LoRAFFNBlock(
                router=self._get_router(layer_idx),
                mlp=mlp,
                lora_experts=lora_experts,
                enable_gradient_checkpointing=self.enable_gradient_checkpointing,
            )
            self.ffn_blocks[layer_idx] = block

            def make_mlp_forward(router_block: LoRAFFNBlock):
                def new_mlp_forward(x: torch.Tensor) -> torch.Tensor:
                    out, aux_loss = router_block.forward(
                        x,
                        attention_mask=self._attention_mask,
                        choices=self._choices,
                        routing_weights=self._routing_weights,
                        routing_detach=self._routing_detach,
                    )
                    if self.training and (aux_loss is not None):
                        self.aux_loss += aux_loss
                    return out

                return new_mlp_forward

            mlp.forward = make_mlp_forward(block)

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass through the LoRA-Mixer model.

        It first resets the auxiliary loss, then passes the inputs to the base model.
        The patched forward methods of the linear layers handle the LoRA mixing.
        Set `capture_routing=True` to store per-layer routing distributions in
        `self.routing_distributions`.

        Shape summary:
            input: matches wrapped model (e.g., input_ids -> (batch_size, seq_len))
            output: matches wrapped model outputs
        """
        capture_routing = kwargs.pop("capture_routing", False)
        routing_detach = kwargs.pop("routing_detach", True)
        self.routing_distributions = None
        self.aux_loss = 0.0
        gini_loss = 0.0
        self._attention_mask = kwargs.get("attention_mask")
        self._choices = (
            []
            if self.training and (self.token_gamma + self.sequence_gamma > 0)
            else None
        )
        self._routing_weights = [] if capture_routing else None
        self._routing_detach = routing_detach
        try:
            outputs = self.wrapped_model(*args, **kwargs)
            if self.training:
                gini_loss = (
                    gini_loss
                    + token_level_consistency_loss(
                        choices=self._choices,
                        attention_mask=self._attention_mask,
                        gamma=self.token_gamma,
                    )
                    if self.token_gamma > 0
                    else gini_loss
                )
                gini_loss = (
                    gini_loss
                    + sequence_level_consistency_loss(
                        choices=self._choices,
                        attention_mask=self._attention_mask,
                        gamma=self.sequence_gamma,
                    )
                    if self.sequence_gamma > 0
                    else gini_loss
                )

        finally:
            self._attention_mask = None
            self._choices = None
            if self._routing_weights is not None:
                self.routing_distributions = torch.stack(self._routing_weights, dim=2)
            self._routing_weights = None

        if self.training:
            self.aux_loss = (
                self.aux_loss / self.num_layers if self.aux_loss > 0 else self.aux_loss
            )
            self.aux_loss = self.aux_loss + gini_loss

        return outputs

    def generate(self, *args, **kwargs):
        """Proxy generation calls to the wrapped causal LM."""
        return self.wrapped_model.generate(*args, **kwargs)

    def generate_batch(self, *args, **kwargs):
        """Proxy dynamic batching generation calls to the wrapped causal LM."""
        if not hasattr(self.wrapped_model, "generate_batch"):
            raise AttributeError("Wrapped model does not expose `generate_batch`.")
        return self.wrapped_model.generate_batch(*args, **kwargs)
