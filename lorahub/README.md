# `lorahub`

This folder contains a LoRAHub-style baseline.

Instead of training a router, it searches for a weighted combination of existing LoRA experts and saves the merged result.

## Main command

```bash
python -m lorahub.train -c path/to/config.json
```

## Config

The config for `lorahub/train.py` should include:

- `model_name_or_path`
- `expert_paths`
- `train_set_configs`
- `output_dir`

Some useful optional fields are:

- `batch_size`
- `max_steps`
- `l1_regularization`
- `weight_bound`
- `seed`
- `save_history`

Each dataset entry inside `train_set_configs` can include:

- `name`
- `fn_path`
- `rl_path`
- `max_num_fn`
- `max_num_rl`
- `max_num_traces`

## Run

```bash
python -m lorahub.train -c configs/lorahub.json
```

## What gets saved

- `output_dir/merged_lora/`
- `output_dir/final_weights.json`
- `output_dir/resolved_config.json`
- `output_dir/optimization_history.json` if history saving is enabled

## Extra package

```bash
pip install nevergrad
```
