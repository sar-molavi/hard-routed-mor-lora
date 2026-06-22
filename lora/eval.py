import json
import re
from fractions import Fraction
import argparse

from datasets import Dataset, logging
from sklearn.metrics import accuracy_score
import numpy as np

from .utils import extract_last_json


def extract_answer(text):
    text = "n/a" if text is None else text
    try:
        answer = extract_last_json(text)
        answer = str(answer["answer"])
    except:
        answer = text
    return answer


def _normalize_label(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.strip().strip('"').strip("'")

    # Extract a simple token if the model returns "answer: X" or "option X".
    match = re.search(r"\b(?:answer|option)\b\s*[:\-]?\s*([a-z0-9]+)", s)
    if match:
        s = match.group(1)

    s = s.strip().strip(".:,;!?)(")

    bool_map = {
        "true": "1",
        "false": "0",
        "yes": "1",
        "no": "0",
        "y": "1",
        "n": "0",
        "t": "1",
        "f": "0",
        "1": "1",
        "0": "0",
    }
    if s in bool_map:
        return bool_map[s]

    grammar_map = {
        "acceptable": "1",
        "unacceptable": "0",
    }
    if s in grammar_map:
        return grammar_map[s]

    return s


def extract_y_true_y_pred(x):
    y_true = _normalize_label(x.get("label"))
    predictions = x.get("predictions") or []
    y_pred_raw = extract_answer(predictions[0]) if predictions else ""
    y_pred = _normalize_label(y_pred_raw)
    return {"y_true": y_true, "y_pred": y_pred}


def score(path):
    """
    Computes and prints the accuracy score for a JSON dataset.
    """
    dataset = Dataset.from_json(path)
    dataset = dataset.map(extract_y_true_y_pred)

    y_pred = list(dataset["y_pred"])
    y_true = list(dataset["y_true"])

    print(accuracy_score(y_true=y_true, y_pred=y_pred))


def normalize_number(s: str) -> Fraction | None:
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "")  # thousands separators
    # Remove trailing punctuation
    s = re.sub(r"[^\d\-./]", "", s)

    try:
        # Fraction can parse "42", "-3", "1/2"
        # It will not parse "42.5" directly unless we convert
        if "." in s and "/" not in s:
            # Convert decimal string to Fraction safely
            return Fraction(s)
        return Fraction(s)
    except Exception:
        return None


def math_score(path) -> None:
    dataset = Dataset.from_json(path)
    dataset = dataset.map(extract_y_true_y_pred_math)

    # Accuracy is simply mean(correct)
    acc = np.mean(dataset["correct"])
    print(f"Accuracy: {acc:.4f}")


def extract_y_true_y_pred_math(x):
    # Raw strings
    y_true_raw = str(x["label"]).strip().lower()
    y_pred_raw = extract_answer(x["predictions"][0])
    y_pred_raw = "" if y_pred_raw is None else y_pred_raw.strip().lower()

    # Normalize to numbers
    y_true = normalize_number(y_true_raw)
    y_pred = normalize_number(y_pred_raw)

    # Exact match after normalization
    correct = int(y_true is not None and y_pred is not None and y_true == y_pred)

    return {
        "correct": correct,
    }


if __name__ == "__main__":
    # Suppress dataset library logs and progress bars for cleaner output
    logging.set_verbosity(logging.CRITICAL)
    logging.disable_progress_bar()

    # CLI arguments
    parser = argparse.ArgumentParser(
        description="Compute accuracy from a JSON dataset."
    )
    parser.add_argument("path", type=str, help="Path to the JSON dataset file.")
    parser.add_argument(
        "--is-math",
        action="store_true",
        help="Flag indicating if the dataset uses math-style answers (with '####').",
    )

    args = parser.parse_args()
    if args.is_math:
        math_score(args.path)
    else:
        score(args.path)
