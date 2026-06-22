import json
import re
from pathlib import Path
from datasets import Dataset


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

ANSWER_PATTERN = re.compile(
    r'"answer"\s*:\s*(?:"([^"]+)"|(-?\d+(?:\.\d+)?))',
    re.IGNORECASE,
)

CODE_FENCE_PATTERN = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences."""
    return CODE_FENCE_PATTERN.sub("", text)


def extract_reasoning(text: str) -> str | None:
    """Return reasoning inside <think>...</think>."""
    match = THINK_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _find_json_objects(text: str):
    """
    Yield JSON objects using balanced brace scanning.
    This is safer than regex for nested or multiline JSON.
    """
    stack = []
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append("{")

        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    yield text[start : i + 1]
                    start = None


def extract_answer(text: str) -> dict | None:
    """
    Extract the last valid JSON object containing 'answer'.
    Works with fenced code blocks and multiline JSON.
    """
    text = strip_code_fences(text)

    last_valid = None

    for candidate in _find_json_objects(text):
        if '"answer"' not in candidate.lower():
            continue

        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, dict) and "answer" in parsed:
            last_valid = parsed

    return last_valid


def strip_final_answer(text: str) -> str | None:
    """
    Remove trailing JSON answer block if present.
    """
    text = strip_code_fences(text)

    matches = list(_find_json_objects(text))
    if not matches:
        return None

    last = matches[-1]
    idx = text.rfind(last)

    return text[:idx].rstrip()


def strip_think_tags(text: str) -> str:
    """
    Remove <think> tags but keep the content.
    """
    return re.sub(r"(</?think>)|(&lt;/?think&gt;)", "", text, flags=re.IGNORECASE)


def extract_answer_at_end(text: str) -> str | None:
    """
    Extract the last answer value even if JSON parsing failed.
    """
    text = strip_code_fences(text)

    matches = list(ANSWER_PATTERN.finditer(text))
    if not matches:
        return None

    last = matches[-1]

    value = last.group(1) if last.group(1) is not None else last.group(2)

    if value is None:
        return None

    return value.strip()


def load_json_dataset(path: str | Path) -> Dataset:
    """Load JSONL dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return Dataset.from_list(records)


def save_json_dataset(dataset: Dataset, path: str | Path) -> None:
    """Save dataset to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")