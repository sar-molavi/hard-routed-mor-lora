"""Reward helpers tailored for the TRL GRPO trainer."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizer

from .utils import extract_last_json


def _extract_final_answer(completion: str) -> str:
    """Pull the final boxed answer if completions contain reasoning traces."""
    final_block = extract_last_json(completion)
    try:
        final_block = final_block["answer"] if final_block else None
    except Exception:
        final_block = None

    if final_block is not None and not isinstance(final_block, str):
        final_block = str(final_block)
    return (final_block or completion).strip()


def compute_reward(
    *,
    correct_reward: float,
    incorrect_reward: float,
    bonus_think: float,
    bonus_json_on_wrong: float,
    ground_truths: list[str],
    completions: list[str],
    prompts: list[str] = None,
    completion_ids: list[list[int]] = None,
    finish_reasons: list[str] = None,
    tokenizer: PreTrainedTokenizer | None = None,
) -> list[float]:
    """
    Reward compatible with TRL's GRPOTrainer: correctness + formatting bonuses.
    """
    if ground_truths is None:
        raise ValueError("`ground_truth` must be provided.")

    rewards: list[float] = []
    final_answers = [_extract_final_answer(text) for text in completions]
    for answer, gt, completion in zip(
        final_answers, ground_truths, completions, strict=True
    ):
        pred = answer.strip().lower()
        target = gt.strip().lower()
        is_correct = pred == target
        base_reward = correct_reward if is_correct else incorrect_reward

        think_start = completion.find("<think>")
        think_end = completion.find("</think>")
        has_think_tags = 0 <= think_start < think_end

        json_block = extract_last_json(completion)
        has_json = json_block is not None

        if has_think_tags:
            bonus = bonus_think
        elif not is_correct and has_json:
            bonus = bonus_json_on_wrong
        else:
            bonus = 0.0

        rewards.append(base_reward + bonus)
    return rewards
