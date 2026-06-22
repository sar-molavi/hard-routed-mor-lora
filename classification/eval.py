from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import PeftModel

from .config import DatasetInfo, FineTuningConfig
from .datasets import ID2LABEL, LABEL2ID, DatasetCollator, get_dataset


@dataclass
class EvalConfig:
    model_id: str
    checkpoint_dir: Path
    output_dir: Path
    datasets: list[DatasetInfo]
    max_length: int
    batch_size: int
    seed: int = 42
    max_eval_samples: int | None = None

    @classmethod
    def from_training_config(
        cls,
        cfg: FineTuningConfig,
        *,
        checkpoint_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        use_validation: bool = True,
        max_eval_samples: int | None = None,
    ) -> "EvalConfig":
        datasets = cfg.validation_dataset_info if use_validation else cfg.dataset_info
        if not datasets:
            raise ValueError("No datasets available for evaluation.")

        resolved_checkpoint_dir = Path(
            checkpoint_dir or os.path.join(cfg.output_dir, "final_checkpoint")
        )
        resolved_output_dir = Path(
            output_dir or os.path.join(cfg.output_dir, "evaluation")
        )
        return cls(
            model_id=cfg.model_id,
            checkpoint_dir=resolved_checkpoint_dir,
            output_dir=resolved_output_dir,
            datasets=datasets,
            max_length=cfg.max_length,
            batch_size=cfg.per_device_eval_batch_size,
            seed=cfg.seed,
            max_eval_samples=max_eval_samples,
        )


def _load_model(model_id: str, checkpoint_dir: Path):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    adapter_file = checkpoint_dir / "adapter_model.safetensors"
    adapter_bin = checkpoint_dir / "adapter_model.bin"
    if adapter_file.exists() or adapter_bin.exists():
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    elif (checkpoint_dir / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))

    model.eval()
    return model


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate_dataset(
    *,
    model,
    tokenizer,
    dataset_info: DatasetInfo,
    max_length: int,
    batch_size: int,
    seed: int,
    max_eval_samples: int | None,
):
    dataset = get_dataset(
        info=[dataset_info],
        max_samples=max_eval_samples,
        seed=seed,
    )
    collator = DatasetCollator(tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    rows: list[dict] = []
    total = 0
    correct = 0
    sample_index = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)

            batch_size_actual = labels.size(0)
            for i in range(batch_size_actual):
                pred_id = int(predictions[i].item())
                label_id = int(labels[i].item())
                pred_label = ID2LABEL[pred_id]
                true_label = ID2LABEL[label_id]
                is_correct = pred_id == label_id
                correct += int(is_correct)
                total += 1
                rows.append(
                    {
                        "dataset": dataset_info.name,
                        "index": sample_index,
                        "text": dataset[sample_index]["text"],
                        "label": true_label,
                        "prediction": pred_label,
                        "prediction_id": pred_id,
                        "label_id": label_id,
                        "correct": is_correct,
                        "logits": logits[i].detach().cpu().tolist(),
                        "probabilities": probabilities[i].detach().cpu().tolist(),
                    }
                )
                sample_index += 1

    accuracy = correct / total if total else 0.0
    return rows, {"dataset": dataset_info.name, "total": total, "correct": correct, "accuracy": accuracy}


def run_evaluation(config: EvalConfig) -> dict:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = _load_model(config.model_id, config.checkpoint_dir)
    if torch.cuda.is_available():
        model = model.cuda()

    summary = {"datasets": [], "total": 0, "correct": 0, "accuracy": 0.0}

    for dataset_info in config.datasets:
        rows, dataset_summary = _evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_info=dataset_info,
            max_length=config.max_length,
            batch_size=config.batch_size,
            seed=config.seed,
            max_eval_samples=config.max_eval_samples,
        )
        output_path = config.output_dir / f"{dataset_info.name}.jsonl"
        _write_jsonl(output_path, rows)

        summary["datasets"].append(
            {
                **dataset_summary,
                "output_path": str(output_path),
            }
        )
        summary["total"] += dataset_summary["total"]
        summary["correct"] += dataset_summary["correct"]

    summary["accuracy"] = (
        summary["correct"] / summary["total"] if summary["total"] else 0.0
    )

    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classification inference/testing.")
    parser.add_argument("-c", "--config", required=True, help="Path to training config JSON.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to the saved checkpoint or adapter directory. Defaults to output_dir/final_checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSONL predictions and summary. Defaults to output_dir/evaluation.",
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Evaluate on validation_dataset_info instead of dataset_info.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on the number of examples per dataset during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FineTuningConfig.from_json(args.config)
    eval_cfg = EvalConfig.from_training_config(
        cfg,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        use_validation=args.use_validation,
        max_eval_samples=args.max_eval_samples,
    )
    summary = run_evaluation(eval_cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
