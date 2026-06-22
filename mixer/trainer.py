"""Custom Hugging Face Trainer for LoRA-Mixer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .utils import (
    LORA_MIXER_WEIGHTS_NAME,
    load_lora_mixer_weights,
    save_lora_mixer_weights,
    unwrap_model,
)


def get_router_temperature(model: torch.nn.Module | None) -> float | None:
    if model is None:
        return None

    base_model = unwrap_model(model)
    mixer_model = getattr(base_model, "mixer_model", base_model)
    temperature = getattr(mixer_model, "gumbel_temperature", None)
    if temperature is not None:
        return float(temperature)

    shared_router = getattr(mixer_model, "shared_router", None)
    if shared_router is not None:
        temperature = getattr(shared_router, "gumbel_temperature", None)
        return None if temperature is None else float(temperature)

    router_layers = getattr(mixer_model, "router_layers", None)
    if isinstance(router_layers, torch.nn.ModuleDict) and router_layers:
        first_router = next(iter(router_layers.values()))
        temperature = getattr(first_router, "gumbel_temperature", None)
        return None if temperature is None else float(temperature)

    return None


class GumbelTemperatureCallback(TrainerCallback):
    """Step Gumbel-Softmax router temperature once per optimizer step."""

    def __init__(
        self,
        *,
        scheduler_name: str,
        initial_temperature: float,
        final_temperature: float,
        hold_steps: float = 0.0,
    ) -> None:
        self.scheduler_name = scheduler_name.lower()
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.hold_steps = float(hold_steps)
        self._validate()

    def _validate(self) -> None:
        if self.scheduler_name not in {"cosine", "exponential"}:
            raise ValueError(
                "Gumbel temperature scheduler must be one of: cosine, exponential."
            )
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")
        if self.final_temperature <= 0:
            raise ValueError("final_temperature must be positive.")
        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature must be <= initial_temperature.")
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative.")

    def _resolve_hold_steps(self, total_steps: int) -> int:
        if self.hold_steps < 1.0:
            return int(total_steps * self.hold_steps)
        return int(self.hold_steps)

    def _temperature_at(self, *, step: int, total_steps: int) -> float:
        total_steps = max(int(total_steps), 1)
        hold_steps = min(self._resolve_hold_steps(total_steps), total_steps)
        anneal_steps = total_steps - hold_steps

        if anneal_steps <= 0 or step >= anneal_steps:
            return self.final_temperature

        progress = min(max(step, 0) / anneal_steps, 1.0)
        if self.scheduler_name == "cosine":
            return self.final_temperature + 0.5 * (
                self.initial_temperature - self.final_temperature
            ) * (1.0 + math.cos(math.pi * progress))

        ratio = self.final_temperature / self.initial_temperature
        return self.initial_temperature * (ratio**progress)

    @staticmethod
    def _unwrap_temperature_target(model: torch.nn.Module) -> torch.nn.Module:
        base_model = unwrap_model(model)
        return getattr(base_model, "mixer_model", base_model)

    def _apply_temperature(
        self,
        *,
        model: torch.nn.Module | None,
        step: int,
        total_steps: int,
    ) -> None:
        if model is None:
            return
        mixer_model = self._unwrap_temperature_target(model)
        if not hasattr(mixer_model, "set_router_temperature"):
            return
        temperature = self._temperature_at(step=step, total_steps=total_steps)
        mixer_model.set_router_temperature(temperature)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        self._apply_temperature(
            model=model,
            step=state.global_step,
            total_steps=state.max_steps,
        )

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        self._apply_temperature(
            model=model,
            step=state.global_step,
            total_steps=state.max_steps,
        )


class BalancedLoRATrainer(Trainer):
    """
    Custom Trainer that injects router auxiliary losses and L2 distillation
    regularization into the standard causal-language-model objective.
    """

    def __init__(
        self,
        *args,
        use_default_loss: bool = True,
        balance_loss_weight: float = 0.1,
        distill_l2_reg: float = 0.0,
        old_expert_params: dict[str, torch.Tensor] | None = None,
        unconstrained_experts: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.balance_loss_weight = balance_loss_weight
        self.use_default_loss = use_default_loss
        self.distill_l2_reg = distill_l2_reg
        self.old_expert_params = old_expert_params
        self.unconstrained_experts = unconstrained_experts or []

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        router_temperature = get_router_temperature(self.model)
        if router_temperature is not None:
            logs["router_temperature"] = router_temperature
        return super().log(logs, start_time=start_time)

    def _compute_task_loss(self, *, model: torch.nn.Module, inputs: dict[str, Any]):
        if "sample_weights" not in inputs:
            outputs = model(**inputs)
            return outputs.loss, outputs

        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab_size = shift_logits.shape[-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        ## Sample-level
        valid_mask = shift_labels.ne(-100).float()
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        loss = (loss * valid_mask).sum(dim=1) / denom

        if sample_weights is None:
            loss = loss.mean()
        else:
            sample_weights = sample_weights.view(-1)
            loss = (loss * sample_weights).sum() / sample_weights.sum()

        return loss, outputs

    def _compute_l2_distill_loss(
        self,
        base_model: torch.nn.Module,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Recompute the L2 penalty against the initial LoRA parameters.

        Returns a fresh tensor every call to avoid accidentally accumulating
        gradients across training steps.
        """
        if self.distill_l2_reg <= 0.0 or self.old_expert_params is None:
            return torch.zeros((), device=device)

        moe_layers = getattr(base_model, "moe_layers", None)
        if not isinstance(moe_layers, torch.nn.ModuleDict):
            return torch.zeros((), device=device)

        current_params = dict(moe_layers.named_parameters())

        masked_indices: torch.Tensor | None = None
        if self.unconstrained_experts:
            num_experts = getattr(base_model, "num_experts", None)
            if isinstance(num_experts, int):
                keep = [
                    idx
                    for idx in range(num_experts)
                    if idx not in self.unconstrained_experts
                ]
                if not keep:
                    return torch.zeros((), device=device)
                masked_indices = torch.tensor(keep, device=device, dtype=torch.long)

        loss = torch.zeros((), device=device)
        for name, old_param in self.old_expert_params.items():
            current_param = current_params.get(name)
            if current_param is None:
                continue

            reference_param = old_param.to(
                device=current_param.device, dtype=current_param.dtype
            )
            delta = current_param - reference_param

            if masked_indices is not None and delta.ndim > 0:
                delta = delta.index_select(0, masked_indices)

            loss = loss + delta.pow(2).sum()

        return loss

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Computes task loss + router auxiliary loss.
        """

        if not self.use_default_loss:
            task_loss, outputs = self._compute_task_loss(model=model, inputs=inputs)
        else:
            inputs.pop("sample_weights", None)
            outputs = model(**inputs)
            task_loss = outputs.loss

        base_model = unwrap_model(model)
        aux_loss = getattr(base_model, "aux_loss", None)

        total_loss = task_loss
        if aux_loss is not None and aux_loss > 0:
            total_loss = total_loss + self.balance_loss_weight * aux_loss

        l2_distill_loss = self._compute_l2_distill_loss(
            base_model, device=total_loss.device
        )
        if self.distill_l2_reg > 0:
            total_loss = total_loss + self.distill_l2_reg * l2_distill_loss

        rank = dist.get_rank() if dist.is_initialized() else 0
        logs = {
            f"task_loss/rank_{rank}": task_loss.detach().item(),
            f"total_loss/rank_{rank}": total_loss.detach().item(),
        }
        if aux_loss is not None:
            logs[f"aux_loss/rank_{rank}"] = (
                aux_loss.detach().item()
                if isinstance(aux_loss, torch.Tensor)
                else float(aux_loss)
            )
        logs[f"l2_distill_loss/rank_{rank}"] = l2_distill_loss.detach().item()
        self.log(logs)

        return (total_loss, outputs) if return_outputs else total_loss

    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:
        """
        Persist only the trainable adapters/routers while keeping HF metadata handling.

        The save operation is gated on `self.args.should_save`, which Hugging Face
        toggles so that only the global process writes when using distributed setups.
        """
        if not self.args.should_save:
            return

        target_dir = Path(output_dir or self.args.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        base_model = unwrap_model(self.model)
        save_lora_mixer_weights(base_model, target_dir)

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str | bool | None,
        model: torch.nn.Module | None = None,
    ):
        """
        Restore router/LoRA parameters from adapter-only checkpoints.
        """

        if isinstance(resume_from_checkpoint, bool):
            candidate = (
                self.state.best_model_checkpoint
                if resume_from_checkpoint and self.state.best_model_checkpoint
                else self.state.last_model_checkpoint
            )
            if candidate is None:
                return
            resume_from_checkpoint = candidate

        checkpoint_dir = Path(resume_from_checkpoint)
        adapter_file = checkpoint_dir / LORA_MIXER_WEIGHTS_NAME

        if adapter_file.is_file():
            base_model = unwrap_model(model or self.model)
            load_lora_mixer_weights(base_model, checkpoint_dir, strict=False)
            return

        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
