"""Convert a mixer_trl GRPO config into mixer/config.py-compatible JSON."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from mixer.config import MixerTrainingConfig


GRPO_ONLY_KEYS = {
    "algorithm",
    "generation",
    "reward",
    "max_prompt_length",
    "max_samples_per_dataset",
    "init_mixer_checkpoint",
}


def _apply_dataset_sample_cap(
    datasets: list[dict[str, Any]],
    *,
    max_samples_per_dataset: int | None,
) -> list[dict[str, Any]]:
    if max_samples_per_dataset is None:
        return datasets

    capped: list[dict[str, Any]] = []
    for entry in datasets:
        row = deepcopy(entry)
        current = row.get("max_num_fn")
        if current is None:
            row["max_num_fn"] = max_samples_per_dataset
        else:
            row["max_num_fn"] = min(int(current), max_samples_per_dataset)
        capped.append(row)
    return capped


def convert_grpo_payload_to_mixer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = set(MixerTrainingConfig.__dataclass_fields__.keys())
    out: dict[str, Any] = {}

    for key, value in payload.items():
        if key in GRPO_ONLY_KEYS:
            continue
        if key in allowed_keys:
            out[key] = deepcopy(value)

    max_samples = payload.get("max_samples_per_dataset")
    out["train_set_configs"] = _apply_dataset_sample_cap(
        deepcopy(payload.get("train_set_configs", [])),
        max_samples_per_dataset=max_samples,
    )
    out["eval_set_configs"] = _apply_dataset_sample_cap(
        deepcopy(payload.get("eval_set_configs", [])),
        max_samples_per_dataset=max_samples,
    )

    if not out["train_set_configs"]:
        raise ValueError("Converted config has no `train_set_configs` entries.")

    return out


def _validate_mixer_payload(payload: dict[str, Any]) -> None:
    with NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    try:
        MixerTrainingConfig.from_json(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def convert_grpo_config_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as handle:
        src = json.load(handle)

    converted = convert_grpo_payload_to_mixer_payload(src)
    if validate:
        _validate_mixer_payload(converted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mixer_trl GRPO config JSON to mixer training config JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to mixer_trl GRPO config JSON.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mixer/config_from_trl.json",
        help="Path to write mixer-compatible config JSON.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation via MixerTrainingConfig.from_json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = convert_grpo_config_file(
        input_path=args.input,
        output_path=args.output,
        validate=not args.no_validate,
    )
    print(f"[INFO] Wrote converted config to {output}")


if __name__ == "__main__":
    main()

