# `lora_offline`

This folder contains our offline RL training pipeline for LoRA experts.

It uses a local FastAPI server backed by vLLM. The training job communicates with that server to generate completions and refresh adapters during training.

## Main run path

The primary entrypoint in this repo is [`launch_offline_rl.sh`](../launch_offline_rl.sh).

The main command is:

```bash
bash launch_offline_rl.sh
```

What the script does:

1. starts `lora_offline.prefetch_rlvf_fastapi_server`
2. waits for the `/health` endpoint
3. launches `accelerate` with `lora_offline.train`

This script is named `launch_offline_rl.sh` because it is not only a trainer wrapper. It launches the server needed by offline RL and then starts the learner. The server is used to generate completions and logprobs while training is running.

Before it is reused, the hard-coded values in [`launch_offline_rl.sh`](../launch_offline_rl.sh) should be edited:

- the working directory
- the GPU ids in `CUDA_VISIBLE_DEVICES`
- the model name
- the config path passed to `-c`

## What is in this folder

- `train.py`: the main offline RL training script
- `prefetch_rlvf_fastapi_server.py`: the local generation server
- `prefetch_rlvf_fastapi_dataset.py`: the async prefetch client
- `vllm_eval.py`: evaluation with vLLM
- `eval.py`: simple scoring of saved predictions

## Extra packages needed here

```bash
pip install bitsandbytes vllm fastapi uvicorn httpx pydantic
```

## Config

The config schema is defined in `lora_offline/config.py`.

The main top-level fields are:

- `model_name_or_path`
- `output_dir`
- `lora_config`
- `prefetch`
- `algorithm`

Inside `prefetch`, the most important fields are:

- `dataset_name`
- `dataset_path`
- `server_url`
- `max_samples`
- `num_generations`
- `max_new_tokens`

Inside `algorithm`, the key field is:

- `loss_type`, which can be `sequence`, `token`, or `token_dapo`

## Manual run without `launch_offline_rl.sh`

First, the server is started:

```bash
python -m lora_offline.prefetch_rlvf_fastapi_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-lora-rank 128
```

Then `prefetch.server_url` in the config should be pointed to that host and port.

After that, training is launched:

```bash
accelerate launch --mixed_precision bf16 \
  -m lora_offline.train \
  -c configs/offline_rl.json
```

The trainer writes checkpoints into `output_dir` and saves the final adapter to `output_dir/final_model`.

## Evaluation

Example:

```bash
python -m lora_offline.vllm_eval \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --lora-path outputs/offline_boolq/final_model \
  --dataset-name boolq \
  --dataset-path data/boolq/test.jsonl \
  --output outputs/offline_boolq_test.jsonl
```

## Scoring predictions

```bash
python -m lora_offline.eval outputs/offline_boolq_test.jsonl
python -m lora_offline.eval outputs/offline_gsm8k_test.jsonl --is-math
```

## Dataset names

The RL dataset loader supports:

- `math`
- `boolq`
- `arc`
- `cola`
- `medical`
- `coding`
- `sst`
- `svamp`
