"""Entry point that wires project configs into TRL's native GRPOTrainer."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from functools import partial
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_trl.config import TRLTrainingConfig
from lora_trl.modeling import load_checkpoint_path, load_model, load_tokenizer
from lora_trl.trainer import build_grpo_trainer
from lora_offline.dataset import get_dataset
from lora_offline.reward import compute_reward as offline_compute_reward


def _build_reward_fn(cfg: TRLTrainingConfig):
    base_reward_fn = partial(
        offline_compute_reward,
        correct_reward=cfg.reward.correct_reward,
        incorrect_reward=cfg.reward.incorrect_reward,
        bonus_think=cfg.reward.bonus_think,
        bonus_json_on_wrong=cfg.reward.bonus_json_on_wrong,
    )

    def reward_fn(
        *,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        ground_truth: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        return base_reward_fn(
            ground_truths=ground_truth,
            completions=completions,
            prompts=prompts,
            completion_ids=completion_ids,
            finish_reasons=kwargs.get("finish_reasons"),
            tokenizer=kwargs.get("tokenizer"),
        )

    return reward_fn


def train(config_path: str) -> None:
    """
    Launch TRL's official GRPO trainer using the curated configuration stack.
    """
    cfg = TRLTrainingConfig.from_json(config_path)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.gradient_checkpointing and world_size > 1:
        warnings.warn(
            "Enabling gradient checkpointing with re-entrant checkpoints disabled "
            "for multi-GPU training."
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    train_dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        dataset_path=cfg.dataset_path,
        tokenizer=tokenizer,
    )
    reward_fn = _build_reward_fn(cfg)

    trainer = build_grpo_trainer(
        cfg=cfg,
        output_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_fn=reward_fn,
    )

    checkpoint = load_checkpoint_path(output_dir)
    trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRL GRPO training for LoRA adapters."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the RL configuration JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
