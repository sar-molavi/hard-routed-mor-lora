"""Reward helpers tailored for the TRL GRPO trainer (correctness only)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def _left_truncate(*arrays: list[torch.Tensor], length: int) -> list[torch.Tensor]:
    return [a[-length:] for a in arrays] if arrays else []


def _right_truncate(*arrays: list[torch.Tensor], length: int) -> list[torch.Tensor]:
    return [a[:length] for a in arrays] if arrays else []


class DataCollatorForRLFinetuning:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_fn: Callable,
        max_length: int | None = None,
        reward_scaling: str | None = None,
        advantage_mode: str = "grpo",
    ):
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.max_length = max_length
        self.reward_scaling = reward_scaling
        self.advantage_mode = advantage_mode

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def __call__(self, batch: list[dict[str | Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        (
            prompts_nested,
            input_token_ids_nested,
            completions_nested,
            generated_token_ids_nested,
            ground_truths_nested,
            token_logprobs_nested,
            finish_reasons_nested,
        ) = self._extract_fields(batch)

        rewards_nested, advantages_nested = self._compute_rewards_and_advantages(
            prompts_nested=prompts_nested,
            completions_nested=completions_nested,
            ground_truths_nested=ground_truths_nested,
            finish_reasons_nested=finish_reasons_nested,
        )

        # prompts = self._flatten(prompts_nested)
        input_token_ids = self._flatten(input_token_ids_nested)
        # completions = self._flatten(completions_nested)
        generated_token_ids = self._flatten(generated_token_ids_nested)
        # ground_truths = self._flatten(ground_truths_nested)
        token_logprobs = self._flatten(token_logprobs_nested)
        finish_reasons = self._flatten(finish_reasons_nested)
        rewards = self._flatten(rewards_nested)
        advantages = self._flatten(advantages_nested)

        # Build model inputs (prompt+completion sequences)
        (input_ids, completion_mask, IS_logprobs) = self._build_model_inputs(
            input_token_ids=input_token_ids,
            generated_token_ids=generated_token_ids,
            token_logprobs=token_logprobs,
        )

        # Convert the finish_reason to binary
        is_stopped = self._convert_finish_reason(finish_reasons)

        padded_tensors, kept_indices = self._left_pad_batch(
            input_ids=input_ids,
            completion_mask=completion_mask,
            IS_logprobs=IS_logprobs,
        )
        if not kept_indices:
            raise RuntimeError("All samples dropped due to padding mismatches.")

        def _select(values):
            return [values[idx] for idx in kept_indices]

        return {
            **padded_tensors,
            "is_stopped": torch.tensor(_select(is_stopped), dtype=torch.bool),
            "rewards": torch.tensor(_select(rewards), dtype=torch.float32),
            "advantages": torch.tensor(_select(advantages), dtype=torch.float32),
        }

    # ------------------------------------------------------------------

    def _compute_rewards_and_advantages(
        self,
        prompts_nested,
        completions_nested,
        ground_truths_nested,
        finish_reasons_nested,
    ):
        rewards_nested = [
            self.reward_fn(
                prompts=prompts,
                completions=completions,
                completion_ids=None,
                ground_truths=ground_truths,
                finish_reasons=finished_reasons,
                tokenizer=self.tokenizer,
            )
            for prompts, completions, ground_truths, finished_reasons in zip(
                prompts_nested,
                completions_nested,
                ground_truths_nested,
                finish_reasons_nested,
            )
        ]

        advantages_nested = [
            self._compute_advantage(rewards) for rewards in rewards_nested
        ]

        return rewards_nested, advantages_nested

    # ------------------------------------------------------------------
    def _compute_advantage(self, rewards):
        mode = (self.advantage_mode or "grpo").lower()
        if mode == "ignore":
            return self._compute_advantage_ignore(rewards)
        if mode == "custom":
            return self._compute_advantage_custom(rewards)
        return self._compute_advantage_grpo(rewards)

    def _compute_advantage_grpo(self, rewards):
        arr = np.asarray(rewards, dtype=np.float32)
        if arr.size == 0:
            return []
        centered = arr - arr.mean()
        if self.reward_scaling == "group" and arr.size > 1:
            std = arr.std()
            if std > 1e-8:
                centered = centered / std
        return centered.tolist()

    def _compute_advantage_ignore(self, rewards):
        arr = np.asarray(rewards, dtype=np.float32)
        return arr.tolist()

    def _compute_advantage_custom(self, rewards):
        arr = np.asarray(rewards, dtype=np.float32)
        if arr.size == 0:
            return []
        pos_arr = arr[arr > 0]
        mean = 0 if len(pos_arr) == 0 else pos_arr.mean()
        centered = arr - mean
        return centered.tolist()

    # ------------------------------------------------------------------
    def _extract_fields(self, batch):
        completions = [ex["generated_text"] for ex in batch]
        generated_token_ids = [ex["generated_token_ids"] for ex in batch]
        finish_reasons = [ex["finish_reason"] for ex in batch]

        ground_truths = [
            [ex["ground_truth"]] * len(ex["generated_text"]) for ex in batch
        ]
        prompts = [[ex["prompt"]] * len(ex["generated_text"]) for ex in batch]
        input_token_ids = [
            [ex["input_token_ids"]] * len(ex["generated_text"]) for ex in batch
        ]

        token_logprobs_nested = [ex["token_logprobs"] for ex in batch]

        return (
            prompts,
            input_token_ids,
            completions,
            generated_token_ids,
            ground_truths,
            token_logprobs_nested,
            finish_reasons,
        )

    # ------------------------------------------------------------------
    def _build_model_inputs(
        self,
        *,
        input_token_ids,
        generated_token_ids,
        token_logprobs,
    ) -> tuple[list[torch.LongTensor], list[torch.LongTensor]]:
        input_ids = []
        completion_mask = []
        IS_logprobs = []

        for prompt_token_ids, completion_token_ids, logprobs in zip(
            input_token_ids, generated_token_ids, token_logprobs
        ):
            token_ids = torch.tensor(
                prompt_token_ids + completion_token_ids, dtype=torch.long
            )
            logprobs = torch.tensor(
                [0] * len(prompt_token_ids) + logprobs, dtype=torch.float32
            )

            input_ids.append(token_ids)
            completion_mask.append(
                self._create_mask_torch(
                    length=len(token_ids), prompt_len=len(prompt_token_ids)
                )
            )
            IS_logprobs.append(logprobs)

        return input_ids, completion_mask, IS_logprobs

    def _create_mask_torch(self, *, length, prompt_len):
        mask = torch.zeros(length, dtype=torch.long)
        mask[prompt_len:] = 1
        return mask

    # ------------------------------------------------------------------
    def _convert_finish_reason(self, finish_reasons):
        return [finish_reason == "stop" for finish_reason in finish_reasons]

    # ------------------------------------------------------------------
    def _flatten(self, nested):
        return [t for row in nested for t in row]

    # ------------------------------------------------------------------
    def _left_pad_batch(
        self,
        *,
        input_ids: list[torch.Tensor],
        completion_mask: list[torch.Tensor],
        IS_logprobs: list[torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[int]]:

        # Determine the target length
        # target_len = max(t.size(0) for t in input_ids)
        # if self.max_length is not None:
        #    target_len = min(target_len, self.max_length)

        target_len = (
            max(t.size(0) for t in input_ids)
            if self.max_length is None
            else self.max_length
        )

        padded_input_ids = []
        padded_completion_mask = []
        padded_logprobs = []
        attention_mask = []
        kept_indices: list[int] = []

        for idx, (ids, c_mask, lp) in enumerate(
            zip(input_ids, completion_mask, IS_logprobs)
        ):
            if not (ids.size(0) == c_mask.size(0) == lp.size(0)):
                logger.warning(
                    f"Dropping sample {idx} due to mismatched lengths (ids={ids}, mask={c_mask.size(0)}, logprobs={lp.size(0)})"
                )
                continue

            # ----- TRUNCATION (truncate if too long) -----
            if ids.size(0) > target_len:
                ids, c_mask, lp = _right_truncate(ids, c_mask, lp, length=target_len)

            # ----- PADDING -----
            pad_len = target_len - ids.size(0)

            padded_ids = torch.cat(
                [
                    torch.full(
                        (pad_len,),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                    ids,
                ]
            )
            padded_input_ids.append(padded_ids)

            padded_cmask = torch.cat(
                [
                    torch.zeros(pad_len, dtype=torch.long),
                    c_mask,
                ]
            )
            padded_completion_mask.append(padded_cmask)

            padded_lp = torch.cat(
                [
                    torch.zeros(pad_len, dtype=torch.float32),
                    lp,
                ]
            )
            padded_logprobs.append(padded_lp)

            att = torch.cat(
                [
                    torch.zeros(pad_len, dtype=torch.bool),
                    torch.ones(ids.size(0), dtype=torch.bool),
                ]
            )
            attention_mask.append(att)
            kept_indices.append(idx)

        if not kept_indices:
            return (
                {
                    "input_ids": torch.empty(0),
                    "completion_mask": torch.empty(0),
                    "attention_mask": torch.empty(0),
                    "token_logprobs": torch.empty(0),
                },
                kept_indices,
            )

        return (
            {
                "input_ids": torch.stack(padded_input_ids),
                "completion_mask": torch.stack(padded_completion_mask),
                "attention_mask": torch.stack(attention_mask),
                "token_logprobs": torch.stack(padded_logprobs),
            },
            kept_indices,
        )
