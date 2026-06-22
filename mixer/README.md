# `mixer`

This is the main folder for HardLoRAMixer itself.

After the individual experts have been trained, this is where the frozen-expert mixer is trained and inference is run.

## Main commands

- `python -m mixer.train -c path/to/config.json`
- `python -m mixer.hf_infer ...`
- `python -m mixer.vllm_eval ...`
- `python -m mixer.count_params -c path/to/config.json`
- `python -m mixer.mixer_trl.train -c path/to/config.json`

## Supervised mixer training

The main training command is:

```bash
python -m mixer.train -c mixer_config.json
```

The config schema is defined in `mixer/config.py`.

The fields that are always needed are:

- `model_name_or_path`
- `expert_paths`
- `enable_lora_attn`
- `normalize_router_weights`
- `top_k`
- `output_dir`
- `train_set_configs`

Some important mixer-specific options are:

- `apply_hard`
- `router_mode`
- `router_shared_across_layers`
- `freeze_router`
- `freeze_experts`
- `distill_l2_reg`
- `balance_loss_weight`

The final weights are saved under:

```text
output_dir/final_model
```

## HF inference

If inference is to be run through regular Hugging Face / PyTorch code, the following is used:

```bash
python -m mixer.hf_infer \
  -c mixer_config.json \
  --dataset-name math \
  --dataset-path data/gsm8k/test.jsonl \
  --output outputs/mixer_gsm8k_preds.jsonl
```

## vLLM inference

If the vLLM path is needed, the following is used:

```bash
python -m mixer.vllm_eval \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --lora-path outputs/mixer_run/final_model \
  --dataset-name boolq \
  --dataset-path data/boolq/test.jsonl \
  --output outputs/mixer_boolq_preds.jsonl
```

## Parameter counting

To inspect trainable parameter counts, the following is used:

```bash
python -m mixer.count_params -c mixer_config.json
```

## TRL-based mixer training

There is also a TRL variant under `mixer_trl/`.

The bundled configs are used as starting templates:

- [`mixer_trl/config.json`](./mixer_trl/config.json)
- [`mixer_trl/config_for_mixer.json`](./mixer_trl/config_for_mixer.json)

Then the following is run:

```bash
python -m mixer.mixer_trl.train -c mixer/mixer_trl/config.json
```

or on multiple GPUs:

```bash
accelerate launch --mixed_precision bf16 \
  -m mixer.mixer_trl.train \
  -c mixer/mixer_trl/config.json
```

## Notes

- The bundled example configs still use old `/nfs/home/...` paths
- `expert_paths` should point to the LoRA weights produced in the first training stage
