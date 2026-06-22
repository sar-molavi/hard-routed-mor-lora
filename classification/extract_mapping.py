from pathlib import Path
import argparse
import json


def extract_predictions(
    eval_dir: str | Path,
    output_path: str | Path,
    adapter_map_path: str | Path | None = None,
) -> dict:
    """
    Reads evaluation JSONL files and writes:

    {
      "dataset_name": {
        "index": {
          "predicted_label": "label",
          "adapter_path": "/path/to/adapter"
        }
      }
    }
    """
    eval_dir = Path(eval_dir)
    output_path = Path(output_path)

    adapter_map = {}
    if adapter_map_path is not None:
        adapter_map_path = Path(adapter_map_path)
        adapter_map = json.loads(adapter_map_path.read_text(encoding="utf-8"))

    result: dict[str, dict[str, dict[str, str | None]]] = {}

    for jsonl_path in sorted(eval_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                dataset_name = row["dataset"]
                index = str(row["index"])
                prediction = row["prediction"]

                result.setdefault(dataset_name, {})[index] = {
                    "predicted_label": prediction,
                    "adapter_path": adapter_map.get(dataset_name),
                }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract predicted labels from evaluation JSONL files."
    )

    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing evaluation .jsonl files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the extracted predictions JSON.",
    )
    parser.add_argument(
        "--adapter-map",
        default=None,
        help="Optional path to adapter_map.json.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extract_predictions(
        eval_dir=args.eval_dir,
        output_path=args.output_path,
        adapter_map_path=args.adapter_map,
    )


if __name__ == "__main__":
    main()