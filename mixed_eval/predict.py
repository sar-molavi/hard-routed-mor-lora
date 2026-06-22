"""Run LoRA-Mixer HF generation on prepared mixed-evaluation prompts."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from mixer.config import MixerTrainingConfig
from mixer.hf_infer import (
    EvalConfig,
    HFEvaluator,
    _finished_output_path,
    _output_already_processed,
    _parse_args as parse_hf_args,
    _unfinished_output_path,
)
from mixer.utils import install_exit_handlers, load_checkpoint_path

from .eval_dataset import get_dataset as get_mixed_dataset
from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


class MixedHFEvaluator(HFEvaluator):
    """HFEvaluator variant that reads already-mixed prompt JSONL files."""

    def _load_dataset(self):
        logger.info("Loading mixed eval dataset from %s", self.config.dataset_path)
        return get_mixed_dataset(
            dataset_path=str(self.config.dataset_path),
            tokenizer=self.tokenizer,
            max_length=self.max_model_len,
        )


def parse_args() -> argparse.Namespace:
    return parse_hf_args()


def main() -> None:
    configure_logging()
    install_exit_handlers()
    args = parse_args()

    output_path = args.output.expanduser()
    logger.info("Starting mixed prediction")
    logger.info("Config path: %s", args.config)
    logger.info("Dataset path: %s", args.dataset_path)
    logger.info("Output path: %s", output_path)
    if _output_already_processed(output_path, resume=args.resume):
        logger.info("Output already processed; exiting")
        return

    logger.info("Loading LoRA-Mixer training config")
    base_cfg = MixerTrainingConfig.from_json(args.config)
    output_root = Path(base_cfg.output_dir).expanduser()
    final_model = output_root / "final_model"
    if final_model.is_dir():
        lora_path = final_model
        logger.info("Using final model from %s", lora_path)
    elif args.require_final_model:
        raise ValueError(f"final_model not found: {final_model}")
    else:
        checkpoint_root = output_root / "checkpoints"
        latest_checkpoint = load_checkpoint_path(checkpoint_root)
        if latest_checkpoint is None:
            raise ValueError("Unable to locate LoRA-Mixer checkpoint in output_dir.")
        lora_path = Path(latest_checkpoint)
        logger.info("Using checkpoint from %s", lora_path)

    expert_paths = [Path(path).expanduser() for path in base_cfg.expert_paths]
    if not expert_paths:
        raise ValueError("Expert paths must be provided in config.")
    if base_cfg.top_k is None:
        raise ValueError("router_top_k must be provided in config.")

    enable_lora_attn = base_cfg.enable_lora_attn
    lora_kwargs = asdict(base_cfg.lora_config) if enable_lora_attn else None

    finished_exists = _finished_output_path(output_path).exists()
    unfinished_exists = _unfinished_output_path(output_path).exists()
    resume = args.resume or finished_exists or unfinished_exists
    logger.info(
        "Runtime options: max_new_tokens=%d num_samples=%d temperature=%s "
        "max_batch_size=%d max_batch_tokens=%d resume=%s",
        args.max_new_tokens,
        args.num_samples,
        args.temperature,
        args.max_batch_size,
        args.max_batch_tokens,
        resume,
    )

    config = EvalConfig(
        model_name_or_path=base_cfg.model_name_or_path,
        lora_path=lora_path,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path.expanduser(),
        output_path=output_path,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        router_top_k=base_cfg.top_k,
        normalize_router_weights=base_cfg.normalize_router_weights,
        router_alpha=0.0,
        router_token_gamma=0.0,
        router_sequence_gamma=0.0,
        jitter_noise=base_cfg.jitter_noise,
        apply_hard=base_cfg.apply_hard,
        router_shared_across_layers=base_cfg.router_shared_across_layers,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        fp16=args.fp16 if args.fp16 else base_cfg.fp16,
        bf16=args.bf16 if args.bf16 else base_cfg.bf16,
        pad_to_multiple_of=args.pad_to_multiple_of,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
        chunk_size=args.chunk_size,
        expert_paths=expert_paths,
        enable_lora_attn=enable_lora_attn,
        lora_kwargs=lora_kwargs,
        num_layers=base_cfg.num_layers,
        max_length=base_cfg.max_length,
        resume=resume,
    )

    logger.info("Initializing HF evaluator and model")
    evaluator = MixedHFEvaluator(config)
    logger.info("Starting generation")
    evaluator.run()
    logger.info("Finished generation; predictions written to %s", output_path)


if __name__ == "__main__":
    main()
