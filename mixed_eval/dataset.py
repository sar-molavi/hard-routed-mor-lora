"""Build mixed GSM8K/BoolQ evaluation prompts."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

from lora_offline.utils import extract_last_after_hashes

from .prompt import (
    MERGED_TWO_OUTPUT_PROMPT_BOOLQ_FIRST,
    MERGED_TWO_OUTPUT_PROMPT_MATH_FIRST,
)
from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


PROMPT_TEMPLATES = {
    "math_first": MERGED_TWO_OUTPUT_PROMPT_MATH_FIRST,
    "boolq_first": MERGED_TWO_OUTPUT_PROMPT_BOOLQ_FIRST,
}
PROMPT_ORDER_CHOICES = (*PROMPT_TEMPLATES.keys(), "random")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    logger.info("Reading JSONL: %s", path)
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
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    logger.info("Writing mixed JSONL: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    logger.info("Wrote %d records to %s", count, path)


def _limit(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 0:
        raise ValueError("Sample limits must be non-negative.")
    return records[:limit]


def _math_question(record: dict[str, Any]) -> str:
    if "question" in record:
        return str(record["question"])
    if "problem" in record:
        return str(record["problem"])
    raise KeyError("Math record must contain either 'question' or 'problem'.")


def _math_answer(record: dict[str, Any]) -> str:
    if "answer" in record:
        answer = extract_last_after_hashes(str(record["answer"]))
        return (answer if answer is not None else str(record["answer"])).strip().lower()
    if "solution" in record:
        # MATH-style solutions do not always use GSM8K's #### marker. Keep the
        # full solution fallback so callers can pre-normalize if needed.
        answer = extract_last_after_hashes(str(record["solution"]))
        return (answer if answer is not None else str(record["solution"])).strip().lower()
    raise KeyError("Math record must contain either 'answer' or 'solution'.")


def _boolq_answer(record: dict[str, Any]) -> str:
    return str(int(bool(record["answer"])))


def _sample_schedule(
    records: list[dict[str, Any]], total: int, rng: random.Random
) -> list[tuple[int, dict[str, Any], bool]]:
    """Shuffle records once, then oversample with replacement after exhaustion."""
    if not records:
        raise ValueError("Cannot sample from an empty dataset.")

    indices = list(range(len(records)))
    rng.shuffle(indices)

    schedule: list[tuple[int, dict[str, Any], bool]] = []
    for out_idx in range(total):
        oversampled = out_idx >= len(indices)
        source_idx = indices[out_idx] if not oversampled else rng.choice(indices)
        schedule.append((source_idx, records[source_idx], oversampled))
    return schedule


def build_mixed_records(
    *,
    math_records: list[dict[str, Any]],
    boolq_records: list[dict[str, Any]],
    seed: int,
    prompt_order: str,
) -> list[dict[str, Any]]:
    """Create paired mixed-task examples with deterministic oversampling."""
    if prompt_order not in PROMPT_ORDER_CHOICES:
        raise ValueError(
            f"Unsupported prompt_order '{prompt_order}'. "
            f"Choose from: {', '.join(sorted(PROMPT_ORDER_CHOICES))}"
        )

    total = max(len(math_records), len(boolq_records))
    if total == 0:
        raise ValueError("Both input datasets are empty.")

    logger.info(
        "Building mixed records: math=%d boolq=%d total=%d seed=%d prompt_order=%s",
        len(math_records),
        len(boolq_records),
        total,
        seed,
        prompt_order,
    )
    rng = random.Random(seed)
    math_schedule = _sample_schedule(math_records, total, rng)
    boolq_schedule = _sample_schedule(boolq_records, total, rng)

    mixed: list[dict[str, Any]] = []
    for index, (math_item, boolq_item) in enumerate(zip(math_schedule, boolq_schedule)):
        math_idx, math_record, math_oversampled = math_item
        boolq_idx, boolq_record, boolq_oversampled = boolq_item
        math_question = _math_question(math_record)
        boolq_question = str(boolq_record["question"])
        passage = str(boolq_record["passage"])
        record_prompt_order = (
            rng.choice(tuple(PROMPT_TEMPLATES))
            if prompt_order == "random"
            else prompt_order
        )
        template = PROMPT_TEMPLATES[record_prompt_order]

        mixed.append(
            {
                "index": index,
                "dataset": "mixed_gsm8k_boolq",
                "prompt_order": record_prompt_order,
                "prompt": template.format(
                    math_question=math_question,
                    passage=passage,
                    boolq_question=boolq_question,
                ),
                "label": {
                    "math_answer": _math_answer(math_record),
                    "boolq_answer": _boolq_answer(boolq_record),
                },
                "source": {
                    "math_index": math_idx,
                    "boolq_index": boolq_idx,
                    "math_oversampled": math_oversampled,
                    "boolq_oversampled": boolq_oversampled,
                },
                "raw": {
                    "math": math_record,
                    "boolq": boolq_record,
                },
            }
        )
    prompt_order_counts = {
        order: sum(1 for record in mixed if record["prompt_order"] == order)
        for order in PROMPT_TEMPLATES
    }
    math_oversampled = sum(1 for record in mixed if record["source"]["math_oversampled"])
    boolq_oversampled = sum(1 for record in mixed if record["source"]["boolq_oversampled"])
    logger.info(
        "Built %d mixed records; prompt_order_counts=%s; oversampled math=%d boolq=%d",
        len(mixed),
        prompt_order_counts,
        math_oversampled,
        boolq_oversampled,
    )
    return mixed


def build_mixed_jsonl(
    *,
    math_path: Path,
    boolq_path: Path,
    output_path: Path,
    seed: int,
    prompt_order: str,
    math_limit: int | None = None,
    boolq_limit: int | None = None,
) -> list[dict[str, Any]]:
    math_records = _limit(_read_jsonl(math_path), math_limit)
    boolq_records = _limit(_read_jsonl(boolq_path), boolq_limit)
    records = build_mixed_records(
        math_records=math_records,
        boolq_records=boolq_records,
        seed=seed,
        prompt_order=prompt_order,
    )
    _write_jsonl(output_path, records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mixed GSM8K/BoolQ JSONL prompts."
    )
    parser.add_argument("--math-path", required=True, type=Path)
    parser.add_argument("--boolq-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-order",
        choices=sorted(PROMPT_ORDER_CHOICES),
        default="random",
    )
    parser.add_argument("--math-limit", type=int, default=None)
    parser.add_argument("--boolq-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    records = build_mixed_jsonl(
        math_path=args.math_path.expanduser(),
        boolq_path=args.boolq_path.expanduser(),
        output_path=args.output.expanduser(),
        seed=args.seed,
        prompt_order=args.prompt_order,
        math_limit=args.math_limit,
        boolq_limit=args.boolq_limit,
    )
    print(f"Wrote {len(records)} mixed prompts to {args.output}")


if __name__ == "__main__":
    main()
