"""Router module aligned with Mixtral-style gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def _checkpoint(function, *args):
    try:
        return checkpoint.checkpoint(function, *args, use_reentrant=False)
    except TypeError:
        return checkpoint.checkpoint(function, *args)


class RSL(nn.Module):
    """Mixtral-style router: linear gate + top-k softmax with optional aux loss."""

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        top_k: int,
        normalize: bool,
        alpha: float = 0.0,
        jitter_noise: float | None = 0.0,
        enable_gradient_checkpointing: bool = False,
        apply_hard: bool | None = None,
        router_mode: str | None = None,
        gumbel_temperature: float = 1.0,
        gumbel_hard: bool = True,
    ):
        """
        Initializes the Router Specialization Layer (RSL).
        """
        super().__init__()
        self.normalize = normalize
        self.alpha = alpha
        self.num_experts = num_experts
        self.top_k = top_k
        self.apply_hard = apply_hard
        self.router_mode = self._resolve_router_mode(router_mode, apply_hard)
        self.gumbel_temperature = float(gumbel_temperature)
        self.gumbel_hard = gumbel_hard
        self.jitter_noise = 0 if jitter_noise is None else jitter_noise
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.gate = nn.Linear(in_features, self.num_experts, bias=False)

    @staticmethod
    def _resolve_router_mode(router_mode: str | None, apply_hard: bool | None) -> str:
        if router_mode is None:
            return "hard_ste" if apply_hard else "soft"

        normalized = router_mode.lower()
        aliases = {
            "hard": "hard_ste",
            "ste": "hard_ste",
            "hard_ste": "hard_ste",
            "soft": "soft",
            "gumbel": "gumbel_softmax",
            "gumbel_softmax": "gumbel_softmax",
        }
        if normalized not in aliases:
            valid = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unknown router_mode={router_mode!r}. Expected one of: {valid}."
            )
        return aliases[normalized]

    def set_gumbel_temperature(self, temperature: float) -> None:
        self.gumbel_temperature = float(temperature)

    def aux_loss(self, *, attention_mask, scores, topk_indices):
        aux_loss = None
        if self.alpha > 0.0:
            balance = 0.0
            mask = None
            valid_count = None
            if attention_mask is not None:
                mask = attention_mask.to(scores.dtype)
                valid_count = mask.sum().clamp(min=1.0)

            # Compute mean router probability per expert
            # (batch_size, seq_len, num_experts) -> (num_experts)
            # mean over batch and seq
            if mask is None:
                Pi = scores.mean(dim=(0, 1))
            else:
                Pi = (scores * mask.unsqueeze(-1)).sum(dim=(0, 1)) / valid_count

            # Compute empirical activation frequency per expert
            # (batch_size, seq_len, k) -> (num_experts)
            if mask is None:
                fi = (
                    # (batch, seq, k) -> (tokens*k, num_experts) -> (num_experts)
                    F.one_hot(topk_indices.reshape(-1), num_classes=self.num_experts)
                    .float()
                    .mean(dim=0)
                )
            else:
                flat_topk = topk_indices.reshape(-1, self.top_k)
                flat_mask = mask.reshape(-1).bool()
                if flat_mask.any():
                    masked_topk = flat_topk[flat_mask]
                    fi = (
                        F.one_hot(
                            masked_topk.reshape(-1),
                            num_classes=self.num_experts,
                        )
                        .float()
                        .mean(dim=0)
                    )
                else:
                    fi = scores.new_zeros(self.num_experts)

            # Balance loss term
            balance = (Pi * fi).sum() * self.num_experts
            balance = balance * self.alpha

            aux_loss = balance
        return aux_loss

    def hard_routing(self, *, scores, topk_indices, topk_weights):

        hard_weights = torch.zeros_like(scores)
        hard_weights.scatter_(dim=-1, index=topk_indices, value=1.0)

        # Straight-through estimator: forward uses hard, backward uses soft.
        router_weights = hard_weights - scores.detach() + scores
        topk_weights = router_weights.gather(dim=-1, index=topk_indices)

        return topk_weights, router_weights

    def soft_routing(self, *, scores, topk_indices, topk_weights):
        # Normalize top-k weights
        # (batch_size, seq_len, k) -> (batch_size, seq_len, k)
        if self.normalize:
            # (batch, seq, k) -> (batch, seq, k)
            topk_weights = F.normalize(topk_weights, p=1, dim=-1)

        if self.top_k < self.num_experts:
            # (batch_size, seq_len, num_experts)
            # create dense top-k routing weights
            router_weights = torch.zeros_like(scores)
            # (batch, seq, num_experts) <- scatter top-k weights
            router_weights.scatter_(
                dim=-1,
                index=topk_indices,
                src=topk_weights,
            )
        else:
            router_weights = scores

        return topk_weights, router_weights

    def gumbel_softmax_routing(self, *, logits):
        temperature = max(float(self.gumbel_temperature), 1e-8)
        if self.training:
            router_weights = F.gumbel_softmax(
                logits.float(),
                tau=temperature,
                hard=self.gumbel_hard,
                dim=-1,
            ).to(logits.dtype)
        else:
            router_weights = F.softmax(logits.float() / temperature, dim=-1).to(
                logits.dtype
            )

        topk_weights, topk_indices = torch.topk(
            router_weights, k=self.top_k, dim=-1
        )
        if self.normalize:
            topk_weights = F.normalize(topk_weights, p=1, dim=-1)

        return topk_weights, topk_indices, router_weights

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.FloatTensor, torch.FloatTensor | None, torch.LongTensor, torch.FloatTensor
    ]:
        """
        Computes expert selection probabilities and the auxiliary balance loss.

        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape
                `(batch_size, seq_len, hidden_dim)`.
            attention_mask (torch.Tensor | None): Optional padding mask of shape
                `(batch_size, seq_len)` where 1 marks valid tokens.

        Returns:
            A tuple containing:
            - scores (torch.FloatTensor): Softmax-normalized probabilities for each
              expert, shape `(batch_size, seq_len, num_experts)`.
            - aux_loss (torch.FloatTensor | None): The scalar auxiliary loss, or None if
              not in training mode or alpha is 0.
            - topk_indices (torch.LongTensor): Indices of the top-k experts for each token,
              shape `(batch_size, seq_len, k)`.
            - topk_weights (torch.FloatTensor): Normalized weights for the top-k experts,
              shape `(batch_size, seq_len, k)`.

        Shape summary:
            input: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)
            output: (batch_size, seq_len, num_experts),
                    scalar aux_loss or None,
                    (batch_size, seq_len, k),
                    (batch_size, seq_len, k)
        """
        def full_forward(input_hidden_states: torch.Tensor):
            # hidden_states: (batch_size, seq_len, hidden_dim)
            old_dtype: torch.dtype = input_hidden_states.dtype
            hidden_states = input_hidden_states

            if self.training and self.jitter_noise > 0:
                # (batch, seq, hidden) * (batch, seq, hidden) -> (batch, seq, hidden)
                hidden_states = hidden_states * torch.empty_like(
                    hidden_states
                ).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

            logits = self.gate(hidden_states)
            # (batch, seq, num_experts) -> (batch, seq, num_experts)
            # logits = logits.float()

            # Compute softmax scores
            # (batch_size, seq_len, num_experts) -> (batch_size, seq_len, num_experts)
            scores = F.softmax(logits, dim=-1)

            if self.router_mode == "gumbel_softmax":
                topk_weights, topk_indices, router_weights = self.gumbel_softmax_routing(
                    logits=logits
                )
            else:
                # Get top-k experts
                # (batch_size, seq_len, num_experts) -> (batch_size, seq_len, k)
                topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

            aux_loss = None
            if self.training:
                aux_loss = self.aux_loss(
                    attention_mask=attention_mask,
                    scores=scores,
                    topk_indices=topk_indices,
                )

            if self.router_mode == "hard_ste":
                topk_weights, router_weights = self.hard_routing(
                    scores=scores, topk_indices=topk_indices, topk_weights=topk_weights
                )
            elif self.router_mode == "soft":
                topk_weights, router_weights = self.soft_routing(
                    scores=scores, topk_indices=topk_indices, topk_weights=topk_weights
                )

            # dtype restore
            router_weights = router_weights.to(old_dtype)
            if aux_loss is not None:
                aux_loss = aux_loss.to(old_dtype)
            topk_weights = topk_weights.to(old_dtype)

            return scores, aux_loss, topk_indices, topk_weights

        if (
            self.enable_gradient_checkpointing
            and self.training
            and hidden_states.requires_grad
        ):
            def checkpoint_forward(input_hidden_states: torch.Tensor):
                scores, aux_loss, topk_indices, topk_weights = full_forward(
                    input_hidden_states
                )
                if aux_loss is None:
                    aux_loss = input_hidden_states.new_zeros(())
                return scores, aux_loss, topk_indices, topk_weights

            scores, aux_loss, topk_indices, topk_weights = _checkpoint(
                checkpoint_forward, hidden_states
            )
        else:
            scores, aux_loss, topk_indices, topk_weights = full_forward(hidden_states)

        return scores, aux_loss, topk_indices, topk_weights


class LoRAExperts(nn.Module):
    """Apply routed LoRA experts for a single projection."""

    def __init__(
        self,
        *,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        lora_scaling: torch.Tensor,
        enable_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(lora_A)
        self.lora_B = nn.Parameter(lora_B)
        self.register_buffer("lora_scaling", lora_scaling.to(lora_A))
        self.num_experts = lora_A.shape[0]
        self.enable_gradient_checkpointing = enable_gradient_checkpointing

    def lora_parameters(self):
        return (self.lora_A, self.lora_B)

    def forward(
        self,
        x: torch.Tensor,
        *,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine routed LoRA experts for a single projection.

        Shape summary:
            input: (batch_size, seq_len, in_features)
            output: (batch_size, seq_len, out_features)
        """

        def lora_forward(
            input_x: torch.Tensor,
            input_topk_weights: torch.Tensor,
            input_topk_indices: torch.Tensor,
        ) -> torch.Tensor:
            """
            Apply the routed LoRA experts to the flattened token stream.

            Shape summary:
                input_x: (batch_size, seq_len, in_features)
                input_topk_weights: (batch_size, seq_len, k)
                input_topk_indices: (batch_size, seq_len, k)
                output: (batch_size, seq_len, out_features)
            """
            batch_size, seq_len, hidden_dim = input_x.shape
            token_count = batch_size * seq_len
            k = input_topk_indices.shape[-1]
            output_dim = self.lora_B.shape[1]

            # (batch, seq, in_features) -> (tokens, in_features)
            flat_x = input_x.reshape(token_count, hidden_dim)
            # (batch, seq, k) -> (tokens, k)
            flat_topk_indices = input_topk_indices.reshape(token_count, k)
            # (batch, seq, k) -> (tokens, k)
            flat_topk_weights = input_topk_weights.reshape(token_count, k)

            # (tokens, out_features)
            final = flat_x.new_zeros((token_count, output_dim))

            for expert_idx in range(self.num_experts):
                # (tokens, k) -> (tokens, k)
                mask = flat_topk_indices == expert_idx
                if not mask.any():
                    continue
                # (tokens, k) -> (selected_tokens,)
                token_idx, k_idx = mask.nonzero(as_tuple=True)
                # (selected_tokens, in_features)
                current_state = flat_x[token_idx]
                lora_A = self.lora_A[expert_idx]
                lora_B = self.lora_B[expert_idx]
                # (selected_tokens, in_features) -> (selected_tokens, rank) -> (selected_tokens, out_features)
                delta = F.linear(current_state, lora_A)
                delta = F.linear(delta, lora_B)
                delta = delta * self.lora_scaling[expert_idx]
                # (selected_tokens,) -> (selected_tokens, 1)
                weights = flat_topk_weights[token_idx, k_idx].unsqueeze(-1)
                # (tokens, out_features) += (selected_tokens, out_features)
                final.index_add_(0, token_idx, delta * weights)

            # (tokens, out_features) -> (batch, seq, out_features)
            return final.reshape(batch_size, seq_len, output_dim)

        if self.enable_gradient_checkpointing and self.training and x.requires_grad:
            lora_out = _checkpoint(lora_forward, x, topk_weights, topk_indices)
        else:
            lora_out = lora_forward(x, topk_weights, topk_indices)

        return lora_out
