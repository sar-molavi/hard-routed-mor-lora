"""Create classification-router inputs from mixed GSM8K/BoolQ prompts."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    logger.info("Reading mixed JSONL: %s", path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    logger.info("Loaded %d mixed records", len(records))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    logger.info("Writing JSONL: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _math_question(record: dict[str, Any]) -> str:
    raw_math = record.get("raw", {}).get("math", {})
    if "question" in raw_math:
        return str(raw_math["question"])
    if "problem" in raw_math:
        return str(raw_math["problem"])
    raise KeyError("Mixed record is missing raw.math.question/raw.math.problem.")


def _boolq_question(record: dict[str, Any]) -> str:
    raw_boolq = record.get("raw", {}).get("boolq", {})
    if "question" not in raw_boolq:
        raise KeyError("Mixed record is missing raw.boolq.question.")
    return str(raw_boolq["question"])


def _ordered_text(
    *,
    math_question: str,
    boolq_question: str,
    prompt_order: str,
    delimiter: str,
) -> str:
    if prompt_order == "math_first":
        return f"{math_question}{delimiter}{boolq_question}"
    if prompt_order == "boolq_first":
        return f"{boolq_question}{delimiter}{math_question}"
    raise ValueError(f"Unsupported prompt_order: {prompt_order}")


def _selected_orders(record: dict[str, Any], mode: str) -> list[str]:
    if mode == "from_mixed":
        order = record.get("prompt_order")
        if order not in {"math_first", "boolq_first"}:
            raise ValueError(f"Mixed record has unsupported prompt_order: {order}")
        return [order]
    if mode == "both":
        return ["math_first", "boolq_first"]
    if mode in {"math_first", "boolq_first"}:
        return [mode]
    raise ValueError(f"Unsupported order mode: {mode}")


def build_classification_bridge(
    *,
    mixed_records: list[dict[str, Any]],
    order_mode: str,
    delimiter: str,
    carrier_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return classifier input rows plus sidecar rows for index mapping."""
    classifier_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    logger.info(
        "Building classification bridge: mixed_records=%d order_mode=%s delimiter=%r",
        len(mixed_records),
        order_mode,
        delimiter,
    )

    for mixed_position, mixed_record in enumerate(mixed_records):
        mixed_index = int(mixed_record.get("index", mixed_position))
        math_question = _math_question(mixed_record)
        boolq_question = _boolq_question(mixed_record)
        label = mixed_record.get("label", {})
        source = mixed_record.get("source", {})

        for prompt_order in _selected_orders(mixed_record, order_mode):
            classifier_index = len(classifier_rows)
            text = _ordered_text(
                math_question=math_question,
                boolq_question=boolq_question,
                prompt_order=prompt_order,
                delimiter=delimiter,
            )
            classifier_rows.append({carrier_key: text})
            manifest_rows.append(
                {
                    "classification_index": classifier_index,
                    "mixed_index": mixed_index,
                    "mixed_position": mixed_position,
                    "classification_prompt_order": prompt_order,
                    "mixed_prompt_order": mixed_record.get("prompt_order"),
                    "math_index": source.get("math_index"),
                    "boolq_index": source.get("boolq_index"),
                    "math_oversampled": source.get("math_oversampled"),
                    "boolq_oversampled": source.get("boolq_oversampled"),
                    "math_answer": label.get("math_answer"),
                    "boolq_answer": label.get("boolq_answer"),
                    "classification_text": text,
                }
            )

    logger.info("Built %d classification rows", len(classifier_rows))
    return classifier_rows, manifest_rows


def write_classification_config(
    *,
    base_config_path: Path,
    output_path: Path,
    carrier_label: str,
    classification_input_path: Path,
) -> None:
    logger.info("Reading base classification config: %s", base_config_path)
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config = deepcopy(config)
    dataset_info = [{"name": carrier_label, "path": str(classification_input_path)}]
    config["dataset_info"] = dataset_info
    config["validation_dataset_info"] = dataset_info
    config["max_train_samples"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote classification config: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create classification/eval.py inputs from mixed eval JSONL."
    )
    parser.add_argument("--mixed-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--base-classification-config",
        type=Path,
        default=None,
        help="Optional base classification config to copy and retarget.",
    )
    parser.add_argument(
        "--carrier-label",
        default="gsm8k",
        choices=["gsm8k", "boolq"],
        help="Dataset label used to make classification/eval.py read the input file.",
    )
    parser.add_argument(
        "--carrier-key",
        default="question",
        help="JSON key expected by the carrier label. Use 'question' for gsm8k/boolq.",
    )
    parser.add_argument(
        "--order-mode",
        default="from_mixed",
        choices=["from_mixed", "math_first", "boolq_first", "both"],
        help="Which connected-question order to emit for classification.",
    )
    parser.add_argument(
        "--delimiter",
        default="_",
        help="Delimiter between connected questions. Default creates {MATH}_{BOOLQ}.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    output_dir = args.output_dir.expanduser()
    classification_input_path = output_dir / "classification_input.jsonl"
    manifest_path = output_dir / "classification_manifest.jsonl"
    config_path = output_dir / "classification_config.json"

    classifier_rows, manifest_rows = build_classification_bridge(
        mixed_records=_read_jsonl(args.mixed_path.expanduser()),
        order_mode=args.order_mode,
        delimiter=args.delimiter,
        carrier_key=args.carrier_key,
    )
    _write_jsonl(classification_input_path, classifier_rows)
    _write_jsonl(manifest_path, manifest_rows)

    if args.base_classification_config is not None:
        write_classification_config(
            base_config_path=args.base_classification_config.expanduser(),
            output_path=config_path,
            carrier_label=args.carrier_label,
            classification_input_path=classification_input_path,
        )

    print(
        json.dumps(
            {
                "classification_input": str(classification_input_path),
                "classification_manifest": str(manifest_path),
                "classification_config": (
                    str(config_path) if args.base_classification_config else None
                ),
                "num_classification_rows": len(classifier_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

