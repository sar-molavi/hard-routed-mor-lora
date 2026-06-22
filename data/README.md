# `data`

This is the local staging area used for datasets.

## What is here now

- `download.ipynb`: an old notebook used for downloading or preparing some datasets

## Expected layout

Most scripts in this repo expect JSONL files somewhere under `data/`, for example:

- `data/gsm8k/train.jsonl`
- `data/gsm8k/test.jsonl`
- `data/boolq/train.jsonl`
- `data/arc-c/train.jsonl`
- `data/cola/train.jsonl`
- `data/medqa/train.jsonl`

The exact filenames are controlled by the config files, so they do not have to match these examples exactly.

## Common raw fields

- GSM8K / math: `question`, `answer`
- BoolQ: `question`, `passage`, `answer`
- ARC: `question`, `choices`, `answerKey`
- CoLA: `sentence`, `label`
- MedQA: `question`, `options`, `answer_idx`
