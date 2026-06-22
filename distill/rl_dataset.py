import argparse
from typing import Any
from pathlib import Path

from datasets import Dataset
from .utils import (
    extract_reasoning,
    extract_answer,
    strip_final_answer,
    extract_answer_at_end,
    strip_think_tags,
    load_json_dataset,
    save_json_dataset,
)


def merge_datasets(datasets: list[Dataset]) -> Dataset:
    merged = {}

    for ds in datasets:
        for row in ds:
            idx = row["index"]

            if idx not in merged:
                # Initialize with shared fields
                merged[idx] = {
                    "index": row["index"],
                    "prompt": row["prompt"],
                    "input_ids": row["input_ids"],
                    "label": row["label"],
                    "source": list(row["source"]),
                    "predictions": list(row["predictions"]),
                    "prediction_lengths": [len(x) for x in row["prediction_token_ids"]],
                    "finish_reasons": list(row["finish_reasons"]),
                }
            else:
                # Concatenate list fields
                merged[idx]["predictions"].extend(row["predictions"])
                merged[idx]["finish_reasons"].extend(row["finish_reasons"])
                merged[idx]["source"].extend(row["source"])
                merged[idx]["prediction_lengths"].extend(
                    [len(x) for x in row["prediction_token_ids"]]
                )

    # Convert back to Dataset
    merged_rows = list(merged.values())
    return Dataset.from_list(merged_rows)


def explode_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
    out = {
        "index": [],
        "prompt": [],
        "input_ids": [],
        "label": [],
        "prediction": [],
        "prediction_length": [],
        "finish_reason": [],
        "source": [],
    }

    for i in range(len(batch["index"])):
        n = len(batch["predictions"][i])

        for j in range(n):
            out["index"].append(batch["index"][i])
            out["prompt"].append(batch["prompt"][i])
            out["input_ids"].append(batch["input_ids"][i])
            out["label"].append(batch["label"][i])
            out["prediction"].append(batch["predictions"][i][j])
            out["finish_reason"].append(batch["finish_reasons"][i][j])
            out["source"].append(batch["source"][i][j])
            out["prediction_length"].append(batch["prediction_lengths"][i][j])

    return out


def _to_str(x):
    try:
        x = str(x)
    except Exception:
        x = None
    return x


def extract_info(example: dict[str, Any]) -> dict[str, Any]:
    prediction = example["prediction"]
    reasoning = answer = cot = response = None

    reasoning = extract_reasoning(prediction)
    if reasoning is None:
        cot = strip_final_answer(prediction)
    if cot is not None:
        cot = strip_think_tags(cot)

    answer = extract_answer(prediction)
    if answer is not None and "answer" in answer:
        answer = answer["answer"]

    if answer is None:
        response = extract_answer_at_end(prediction)

    answer = _to_str(answer)
    response = _to_str(response)

    return {
        "reasoning": reasoning,
        "answer": answer,
        "cot": cot,
        "response": response,
    }


def filter_unfinished(example: dict[str, Any]) -> bool:
    return example["finish_reason"] == "stop"


def filter_incorrect_and_empty_reasoning(example: dict[str, Any]) -> bool:
    prediction = (
        example["answer"] if example["answer"] is not None else example["response"]
    )
    reasoning = example["reasoning"] or example["cot"]

    if prediction is None or reasoning is None:
        return False
    
    if not reasoning.strip():
        return False

    label = example["label"].strip().lower()
    prediction = str(prediction).strip().lower()

    return label == prediction


def collapse_dataset(expanded_dataset: Dataset) -> Dataset:
    collapsed = {}

    for row in expanded_dataset:
        idx = row["index"]

        if idx not in collapsed:
            collapsed[idx] = {
                "index": row["index"],
                "prompt": row["prompt"],
                "input_ids": row["input_ids"],
                "label": row["label"],
                "predictions": [row["prediction"]],
                "answers": [row.get("answer")],
                "responses": [row.get("response")],
                "reasonings": [row.get("reasoning")],
                "cots": [row.get("cot")],
                "sources": [row.get("source")],
                "prediction_lengths": [row.get("prediction_length")],
            }
        else:
            collapsed[idx]["predictions"].append(row["prediction"])
            collapsed[idx]["answers"].append(row.get("answer"))
            collapsed[idx]["responses"].append(row.get("response"))
            collapsed[idx]["reasonings"].append(row.get("reasoning"))
            collapsed[idx]["cots"].append(row.get("cot"))
            collapsed[idx]["sources"].append(row.get("source"))
            collapsed[idx]["prediction_lengths"].append(row.get("prediction_length"))

    return Dataset.from_list(list(collapsed.values()))


def pipeline(path: str | Path) -> Dataset:
    path = Path(path)
    data_files = [x for x in path.glob("*.jsonl") if x.stem != "collapsed"]

    datasets = [load_json_dataset(p) for p in data_files]
    if len(datasets) == 0:
        raise ValueError("There is not JSONL.")

    datasets = [
        ds.map(lambda x: {"source": [p.stem] * len(x["predictions"])})
        for ds, p in zip(datasets, data_files)
    ]

    keep = [
        "index",
        "prompt",
        "input_ids",
        "label",
        "predictions",
        "finish_reasons",
        "source",
        "prediction_token_ids",
    ]
    datasets = [ds.select_columns(keep) for ds in datasets]

    dataset = merge_datasets(datasets)

    dataset = dataset.map(
        explode_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.filter(filter_unfinished)
    dataset = dataset.map(extract_info)
    dataset = dataset.filter(filter_incorrect_and_empty_reasoning)
    dataset = collapse_dataset(dataset)

    save_json_dataset(dataset, path / "collapsed.jsonl")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    pipeline(args.path)
