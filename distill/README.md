# `distill`

This folder contains small helper scripts used to turn generation outputs into cleaner distilled reasoning datasets.

## Scripts

- `python -m distill.dataset path/to/prediction_dir`
- `python -m distill.rl_dataset path/to/prediction_dir`

Both scripts read JSONL prediction files from a directory, extract usable reasoning traces, and save a collapsed dataset back into the same directory.

## Expected input

The JSONL files should contain fields like:

- `index`
- `prompt`
- `input_ids`
- `label`
- `predictions`
- `finish_reasons`
- `prediction_token_ids`

These are the same kinds of records written by the evaluation scripts in this repo.

## Run

```bash
python -m distill.dataset outputs/generations_dir
```

or

```bash
python -m distill.rl_dataset outputs/generations_dir
```

Both scripts create:

```text
path/to/prediction_dir/collapsed.jsonl
```