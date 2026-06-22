"""Score mixed GSM8K/BoolQ generation outputs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from lora_offline.utils import extract_last_json
from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


@dataclass
class Counts:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _normalize(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _normalize_math(value: Any) -> str:
    text = _normalize(value)
    if not text:
        return text

    # Remove common formatting noise before numeric comparison. Decimal keeps
    # values like 10 and 10.0 equal without introducing float roundoff.
    numeric_text = text.replace(",", "")
    try:
        number = Decimal(numeric_text)
    except InvalidOperation:
        return text

    if number == number.to_integral_value():
        return str(number.to_integral_value())
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def _extract_prediction(prediction: str) -> dict[str, Any]:
    parsed = extract_last_json("n/a" if prediction is None else prediction)
    return parsed if isinstance(parsed, dict) else {}


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    logger.info("Scoring %d mixed prediction records", len(records))
    math = Counts()
    boolq = Counts()
    joint = Counts()
    scored_records: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Scoring mixed predictions", unit="record"):
        label = record.get("label") or {}
        prediction_text = (record.get("predictions") or [""])[0]
        parsed = _extract_prediction(prediction_text)

        math_true = _normalize_math(label.get("math_answer"))
        boolq_true = _normalize(label.get("boolq_answer"))
        math_pred = _normalize_math(parsed.get("math_answer"))
        boolq_pred = _normalize(parsed.get("boolq_answer"))

        math_match = math_true == math_pred
        boolq_match = boolq_true == boolq_pred
        joint_match = math_match and boolq_match

        math.total += 1
        boolq.total += 1
        joint.total += 1
        math.correct += int(math_match)
        boolq.correct += int(boolq_match)
        joint.correct += int(joint_match)

        scored = dict(record)
        scored["extracted_prediction"] = {
            "math_answer": math_pred,
            "boolq_answer": boolq_pred,
        }
        scored["matched"] = {
            "math": math_match,
            "boolq": boolq_match,
            "joint": joint_match,
        }
        scored_records.append(scored)

    summary = {
        "math": {
            "correct": math.correct,
            "total": math.total,
            "accuracy": math.accuracy,
        },
        "boolq": {
            "correct": boolq.correct,
            "total": boolq.total,
            "accuracy": boolq.accuracy,
        },
        "joint": {
            "correct": joint.correct,
            "total": joint.total,
            "accuracy": joint.accuracy,
        },
        "records": scored_records,
    }
    logger.info(
        "Finished scoring: math=%d/%d boolq=%d/%d joint=%d/%d",
        math.correct,
        math.total,
        boolq.correct,
        boolq.total,
        joint.correct,
        joint.total,
    )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    logger.info("Reading predictions JSONL: %s", path)
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
    logger.info("Loaded %d prediction records", len(records))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    logger.info("Writing scored JSONL: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d scored records to %s", len(records), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed GSM8K/BoolQ generation JSONL."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scored-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    logger.info("Starting mixed prediction evaluation")
    summary = score_records(_read_jsonl(args.predictions.expanduser()))
    records = summary.pop("records")

    if args.scored_output is not None:
        _write_jsonl(args.scored_output.expanduser(), records)
    if args.summary_output is not None:
        logger.info("Writing summary JSON: %s", args.summary_output)
        args.summary_output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.expanduser().write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote summary JSON")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
