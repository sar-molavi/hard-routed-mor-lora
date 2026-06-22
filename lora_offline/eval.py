import json
import argparse
import re

from datasets import Dataset, logging
from sklearn.metrics import accuracy_score

from .utils import extract_last_json


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_answer(text):
    text = "n/a" if text is None else text
    try:
        answer = extract_last_json(text)
        answer = str(answer["answer"])
    except:
        answer = text
    return answer


def normalize_math_answer(text):
    """
    Normalize GSM8K-style answers to comparable integer strings.

    Predictions may contain reasoning, JSON answer blocks, or a final
    '#### <answer>' marker. GSM8K gold answers are integers, so the last
    numeric value is used after removing formatting such as commas.
    """
    text = "" if text is None else str(text).strip().lower()
    if not text:
        return ""

    hash_matches = re.findall(r"####\s*(.+)", text)
    if hash_matches:
        text = hash_matches[-1]

    matches = NUMBER_RE.findall(text)
    if not matches:
        return text.strip()

    answer = matches[-1].replace(",", "")
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")
    return answer


def extract_y_true_y_pred_math(x):
    """
    Extracts ground truth (y_true) and predicted (y_pred) labels for
    math reasoning datasets, where answers are embedded in text after '####'.
    """
    y_true = normalize_math_answer(x["label"])
    y_pred = normalize_math_answer(extract_answer(x["predictions"][0]))

    return {"y_true": y_true, "y_pred": y_pred}


def extract_y_true_y_pred(x):
    """
    Extracts ground truth (y_true) and predicted (y_pred) labels for
    standard classification datasets, where labels are direct strings.
    """
    y_true = str(x["label"]).strip().lower()
    y_pred = extract_answer(x["predictions"][0])
    y_pred = "" if y_pred is None else y_pred.strip().lower()

    return {"y_true": y_true, "y_pred": y_pred}


def score(path, is_math=False):
    """
    Computes and prints the accuracy score for a JSON dataset.
    """
    dataset = Dataset.from_json(path)
    dataset = dataset.map(
        extract_y_true_y_pred_math if is_math else extract_y_true_y_pred
    )

    y_pred = list(dataset["y_pred"])
    y_true = list(dataset["y_true"])

    print(accuracy_score(y_true=y_true, y_pred=y_pred))


if __name__ == "__main__":
    # Suppress dataset library logs and progress bars for cleaner output
    logging.set_verbosity(logging.CRITICAL)
    logging.disable_progress_bar()

    # CLI arguments
    parser = argparse.ArgumentParser(
        description="Compute accuracy from a JSON dataset."
    )
    parser.add_argument(
        "path", type=str, help="Path to the JSON dataset file."
    )
    parser.add_argument(
        "--is-math",
        action="store_true",
        help="Flag indicating if the dataset uses math-style answers (with '####').",
    )

    args = parser.parse_args()
    score(args.path, args.is_math)
