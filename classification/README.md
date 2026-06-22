# `classification`

This folder contains the classifier baseline used as a router.

Instead of generating an answer directly, this model predicts which task family an input belongs to.

## Main commands

- `python -m classification.train -c classification/config.json`
- `python -m classification.eval -c classification/config.json --use-validation`
- `python -m classification.extract_mapping ...`

## What it predicts

The classifier predicts one of these labels:

- `medqa`
- `gsm8k`
- `cola`
- `arc`
- `boolq`

It uses only the input text, not the full generative prompt.

## Config

The config schema is in `classification/config.py`.

The bundled [`config.json`](./config.json) is just a starting point. The dataset paths should be edited before it is used.

The main fields are:

- `model_id`
- `output_dir`
- `dataset_info`
- `validation_dataset_info`
- `max_length`
- `per_device_train_batch_size`
- `per_device_eval_batch_size`

## Train

```bash
python -m classification.train -c classification/config.json
```

The final checkpoint is saved to:

```text
output_dir/final_checkpoint
```

## Evaluate

```bash
python -m classification.eval \
  -c classification/config.json \
  --checkpoint-dir outputs/bert_cls/final_checkpoint \
  --output-dir outputs/bert_cls/eval \
  --use-validation
```

This writes one JSONL file per dataset and also saves a `summary.json`.

## Where we use it

This baseline is mainly used together with `mixed_eval.classification_bridge` and `mixed_eval.classification_route_map`.
