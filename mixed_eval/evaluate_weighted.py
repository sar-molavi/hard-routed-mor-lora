"""Weighted mixed evaluation that de-biases oversampled source examples."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .evaluate import _extract_prediction, _normalize, _normalize_math
from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


@dataclass
class WeightedCounts:
    correct: float = 0.0
    total: float = 0.0
    unique: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    logger.info("Reading JSONL: %s", path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    logger.info("Loaded %d rows", len(rows))
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    logger.info("Writing weighted scored JSONL: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d scored records", len(records))


def _source_from_record(
    record: dict[str, Any],
    *,
    mixed_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    source = (record.get("routing") or {}).get("source")
    if source:
        return source

    source = record.get("source")
    if source:
        return source

    record_index = record.get("index")
    if record_index is not None:
        mixed = mixed_by_index.get(int(record_index))
        if mixed is not None:
            return mixed.get("source") or {}

    return {}


def _source_key(source: dict[str, Any], task: str, fallback_index: int) -> str:
    key_name = f"{task}_index"
    value = source.get(key_name)
    if value is None:
        return f"missing:{fallback_index}"
    return str(value)


def _pair_key(source: dict[str, Any], fallback_index: int) -> str:
    math_index = source.get("math_index")
    boolq_index = source.get("boolq_index")
    if math_index is None or boolq_index is None:
        return f"missing:{fallback_index}"
    return f"{math_index}::{boolq_index}"


def _prepare_scored_records(
    records: list[dict[str, Any]],
    *,
    mixed_by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    scored_records: list[dict[str, Any]] = []
    for fallback_index, record in enumerate(
        tqdm(records, desc="Extracting predictions", unit="record")
    ):
        label = record.get("label") or {}
        prediction_text = (record.get("predictions") or [""])[0]
        parsed = _extract_prediction(prediction_text)
        source = _source_from_record(record, mixed_by_index=mixed_by_index)

        math_true = _normalize_math(label.get("math_answer"))
        boolq_true = _normalize(label.get("boolq_answer"))
        math_pred = _normalize_math(parsed.get("math_answer"))
        boolq_pred = _normalize(parsed.get("boolq_answer"))

        math_match = math_true == math_pred
        boolq_match = boolq_true == boolq_pred

        scored = dict(record)
        scored["source"] = source
        scored["source_keys"] = {
            "math": _source_key(source, "math", fallback_index),
            "boolq": _source_key(source, "boolq", fallback_index),
            "pair": _pair_key(source, fallback_index),
        }
        scored["extracted_prediction"] = {
            "math_answer": math_pred,
            "boolq_answer": boolq_pred,
        }
        scored["matched"] = {
            "math": math_match,
            "boolq": boolq_match,
            "joint": math_match and boolq_match,
        }
        scored_records.append(scored)
    return scored_records


def _weighted_counts(
    records: list[dict[str, Any]],
    *,
    key_name: str,
    match_name: str,
) -> WeightedCounts:
    key_counts = Counter(record["source_keys"][key_name] for record in records)
    counts = WeightedCounts(unique=len(key_counts))

    for record in records:
        key = record["source_keys"][key_name]
        weight = 1.0 / key_counts[key]
        counts.total += weight
        counts.correct += weight * float(record["matched"][match_name])
        record.setdefault("weights", {})[match_name] = weight

    return counts


def score_weighted_records(
    records: list[dict[str, Any]],
    *,
    mixed_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mixed_by_index = {
        int(record["index"]): record
        for record in (mixed_records or [])
        if "index" in record
    }
    scored_records = _prepare_scored_records(records, mixed_by_index=mixed_by_index)

    math = _weighted_counts(scored_records, key_name="math", match_name="math")
    boolq = _weighted_counts(scored_records, key_name="boolq", match_name="boolq")
    joint = _weighted_counts(scored_records, key_name="pair", match_name="joint")

    summary = {
        "math": {
            "weighted_correct": math.correct,
            "weighted_total": math.total,
            "unique_total": math.unique,
            "accuracy": math.accuracy,
        },
        "boolq": {
            "weighted_correct": boolq.correct,
            "weighted_total": boolq.total,
            "unique_total": boolq.unique,
            "accuracy": boolq.accuracy,
        },
        "joint": {
            "weighted_correct": joint.correct,
            "weighted_total": joint.total,
            "unique_total": joint.unique,
            "accuracy": joint.accuracy,
        },
        "records": scored_records,
    }
    logger.info(
        "Weighted accuracy: math=%.6f boolq=%.6f joint=%.6f",
        math.accuracy,
        boolq.accuracy,
        joint.accuracy,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weighted evaluation for mixed GSM8K/BoolQ predictions."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--mixed-path",
        type=Path,
        default=None,
        help="Optional original mixed JSONL. Used if prediction rows lack source indices.",
    )
    parser.add_argument("--scored-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    mixed_records = (
        _read_jsonl(args.mixed_path.expanduser()) if args.mixed_path else None
    )
    summary = score_weighted_records(
        _read_jsonl(args.predictions.expanduser()),
        mixed_records=mixed_records,
    )
    records = summary.pop("records")

    if args.scored_output is not None:
        _write_jsonl(args.scored_output.expanduser(), records)
    if args.summary_output is not None:
        logger.info("Writing weighted summary JSON: %s", args.summary_output)
        args.summary_output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.expanduser().write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

