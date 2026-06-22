"""Map classification-router predictions back to mixed prompts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .logging_utils import configure_logging


logger = logging.getLogger(__name__)

DEFAULT_ID2LABEL = {
    0: "medqa",
    1: "gsm8k",
    2: "cola",
    3: "arc",
    4: "boolq",
}


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
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    logger.info("Writing JSONL: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(records), path)


def _load_adapter_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    logger.info("Reading adapter map: %s", path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): str(value) for key, value in payload.items()}


def _probabilities_by_label(row: dict[str, Any]) -> dict[str, float]:
    probabilities = row.get("probabilities") or []
    return {
        DEFAULT_ID2LABEL[index]: float(prob)
        for index, prob in enumerate(probabilities)
        if index in DEFAULT_ID2LABEL
    }


def _top_labels(probs: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"label": label, "probability": prob}
        for label, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True)
    ]


def build_routing_manifest(
    *,
    mixed_records: list[dict[str, Any]],
    classification_manifest: list[dict[str, Any]],
    classification_predictions: list[dict[str, Any]],
    adapter_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Join classifier outputs to mixed records by classification row index."""
    logger.info(
        "Building routing manifest: mixed=%d manifest=%d predictions=%d",
        len(mixed_records),
        len(classification_manifest),
        len(classification_predictions),
    )
    predictions_by_index = {
        int(row["index"]): row for row in classification_predictions
    }
    mixed_by_position = {idx: row for idx, row in enumerate(mixed_records)}

    output_rows: list[dict[str, Any]] = []
    for manifest_row in tqdm(
        classification_manifest,
        desc="Mapping classifier routes",
        unit="record",
    ):
        classification_index = int(manifest_row["classification_index"])
        prediction = predictions_by_index.get(classification_index)
        if prediction is None:
            raise KeyError(
                f"Missing classification prediction for index={classification_index}"
            )

        mixed_position = int(manifest_row["mixed_position"])
        mixed_record = mixed_by_position[mixed_position]
        probs = _probabilities_by_label(prediction)
        ranked = _top_labels(probs)
        predicted_label = str(prediction.get("prediction", ""))

        output_rows.append(
            {
                "mixed_index": manifest_row["mixed_index"],
                "mixed_position": mixed_position,
                "classification_index": classification_index,
                "classification_prompt_order": manifest_row[
                    "classification_prompt_order"
                ],
                "mixed_prompt_order": manifest_row.get("mixed_prompt_order"),
                "predicted_label": predicted_label,
                "predicted_adapter": adapter_map.get(predicted_label),
                "ranked_labels": ranked,
                "adapter_candidates": [
                    {
                        "label": item["label"],
                        "probability": item["probability"],
                        "adapter": adapter_map.get(item["label"]),
                    }
                    for item in ranked
                ],
                "source": {
                    "math_index": manifest_row.get("math_index"),
                    "boolq_index": manifest_row.get("boolq_index"),
                    "math_oversampled": manifest_row.get("math_oversampled"),
                    "boolq_oversampled": manifest_row.get("boolq_oversampled"),
                },
                "label": mixed_record.get("label"),
                "prompt": mixed_record.get("prompt"),
                "classification_text": manifest_row.get("classification_text"),
                "classifier": {
                    "label": prediction.get("label"),
                    "prediction_id": prediction.get("prediction_id"),
                    "label_id": prediction.get("label_id"),
                    "correct": prediction.get("correct"),
                    "logits": prediction.get("logits"),
                    "probabilities": prediction.get("probabilities"),
                },
            }
        )

    logger.info("Built %d routing rows", len(output_rows))
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join classification/eval.py outputs to mixed eval prompts."
    )
    parser.add_argument("--mixed-path", required=True, type=Path)
    parser.add_argument("--classification-manifest", required=True, type=Path)
    parser.add_argument("--classification-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--adapter-map",
        type=Path,
        default=None,
        help="Optional JSON map from classifier label to LoRA adapter path/name.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    rows = build_routing_manifest(
        mixed_records=_read_jsonl(args.mixed_path.expanduser()),
        classification_manifest=_read_jsonl(args.classification_manifest.expanduser()),
        classification_predictions=_read_jsonl(
            args.classification_predictions.expanduser()
        ),
        adapter_map=_load_adapter_map(args.adapter_map.expanduser() if args.adapter_map else None),
    )
    _write_jsonl(args.output.expanduser(), rows)
    print(json.dumps({"routing_manifest": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

