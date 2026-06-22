# `lora_trl`

This folder contains our TRL-based GRPO training code for a single LoRA expert.

This is used when RL-style training is needed without the separate offline server flow from `lora_offline/`.

## Main command

```bash
python -m lora_trl.train -c lora_trl/config.json
```

For multi-GPU runs, the following is used:

```bash
accelerate launch --mixed_precision bf16 \
  -m lora_trl.train \
  -c path/to/config.json
```

## Config

The schema is defined in `lora_trl/config.py`.

The bundled [`config.json`](./config.json) is only an example. It still contains old paths, so it should be copied and edited first.

The main config sections are:

- model settings
- dataset settings
- trainer settings
- `algorithm` for GRPO options
- `generation` for sampling
- `reward` for reward shaping

## Supported datasets

This code reuses `lora_offline.dataset.get_dataset`, so the supported dataset names include:

- `math`
- `boolq`
- `arc`
- `cola`
- `medical`
- `coding`
- `sst`
- `svamp`

## Output

The trainer resumes from the latest checkpoint inside `output_dir` and writes the final adapter to:

```text
output_dir/final_model
```

## Notes

- GRPO needs `algorithm.num_generations >= 2`
- `dataset_path` can be a JSONL file or a directory of JSONL files
