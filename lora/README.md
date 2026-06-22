# `lora`

This folder contains our standard supervised LoRA and QLoRA training code.

If a single task expert is to be trained without RL, this is usually the first place used.

## What is run here

For training:

```bash
python -m lora.train -c path/to/config.json
```

For vLLM-based evaluation:

```bash
python -m lora.vllm_eval --model-name ... --lora-path ... --dataset-name ... --dataset-path ... --output ...
```

For simple scoring of saved predictions:

```bash
python -m lora.eval path/to/predictions.jsonl
```

## Datasets this code knows about

The formatter in `lora.dataset` currently supports:

- `math`
- `math_no_reasoning`
- `cola`
- `arc`
- `medqa`
- `boolq`

The raw JSONL fields expected here are:

- `math`: `question`, `answer`
- `cola`: `sentence`, `label`
- `arc`: `question`, `choices`, `answerKey`
- `medqa`: `question`, `options`, `answer_idx`
- `boolq`: `question`, `passage`, `answer`

## Config

The config schema is defined in `lora/config.py`.

The fields typically set are:

- `model_name_or_path`
- `use_lora`
- `use_qlora`
- `lora_config`
- `dataset_name`
- `dataset_path`
- `output_dir`

Useful optional fields include:

- `validation_dataset_path`
- `deepspeed_config_path`
- `bf16` or `fp16`
- `gradient_checkpointing`
- `load_best_model_at_end`

## Example training command

```bash
python -m lora.train -c configs/lora_math.json
```

If multi-GPU training is needed, something like the following is used:

```bash
torchrun --nproc_per_node=4 -m lora.train -c configs/lora_math.json
```

The script writes `checkpoint-*` folders inside `output_dir` and resumes from the latest one if it is run again.

## Example evaluation command

```bash
python -m lora.vllm_eval \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --lora-path outputs/gsm8k/final_checkpoint \
  --dataset-name math \
  --dataset-path data/gsm8k/test.jsonl \
  --output outputs/gsm8k_test_preds.jsonl
```

If `--lora-path` is omitted, the base model is run without the adapter.

## Scoring predictions

For non-math tasks:

```bash
python -m lora.eval outputs/boolq_preds.jsonl
```

For GSM8K-style math evaluation:

```bash
python -m lora.eval outputs/gsm8k_preds.jsonl --is-math
```