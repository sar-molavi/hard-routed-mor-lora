# HardLoRAMixer

This repository contains the code for our paper [HardLoRAMixer.pdf](./HardLoRAMixer.pdf), _Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs_.

First, a set of task-specific LoRA experts is trained. Then those experts are frozen and a mixer is trained to route between them. The repository also contains the extra code used for classifier routing, mixed-task evaluation, LoRAHub-style merging, and distillation.

## What is in each folder

- `lora/`: regular supervised LoRA and QLoRA training code
- `lora_offline/`: offline RL training code
- `lora_trl/`: TRL-based GRPO code for single LoRA experts
- `mixer/`: the main HardLoRAMixer training and inference code
- `classification/`: the classifier baseline used as a router
- `mixed_eval/`: mixed GSM8K + BoolQ evaluation scripts
- `lorahub/`: a LoRAHub-style baseline
- `distill/`: small helper scripts for building distilled reasoning datasets
- `data/`: local dataset staging

A small `README.md` has been added inside each important folder so that it is easier to see what belongs where.

## How the environment is usually set up

There is no pinned `requirements.txt`, so the environment is usually initialized with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets peft accelerate safetensors tqdm numpy
pip install scikit-learn evaluate
```

Then additional packages are installed depending on what needs to be run:

- `pip install bitsandbytes` for QLoRA
- `pip install trl` for TRL / GRPO training
- `pip install vllm` for vLLM inference and server-based generation
- `pip install fastapi uvicorn httpx pydantic` for the offline RL server
- `pip install nevergrad` for the LoRAHub baseline

If training is done on multiple GPUs, `accelerate` or `torchrun` should also be configured on the machine.

## General notes

- Everything is run from the repository root with `python -m ...`
- Most training scripts take a JSON config through `-c` or `--config`
- Some example configs still contain old absolute `/nfs/home/...` paths, so they should be edited before reuse
- Most training scripts resume automatically from the latest `checkpoint-*` folder in `output_dir`

## Intended workflow

### 1. Train single experts

For supervised LoRA training, the following is used:

```bash
python -m lora.train -c path/to/config.json
```

For TRL-based GRPO training of a single expert, the following is used:

```bash
python -m lora_trl.train -c path/to/config.json
```

For offline RL training, the main entrypoint in this repository is:

```bash
bash launch_offline_rl.sh
```

That script is the launcher for the full offline RL run. It starts the local FastAPI/vLLM server first, waits for the `/health` check to pass, and then launches `lora_offline.train` with `accelerate`. In other words, it initializes the generation server that the offline RL trainer depends on and then starts the learner.

### 2. Train the mixer

For the supervised mixer stage, the following is used:

```bash
python -m mixer.train -c path/to/mixer_config.json
```

For the TRL-based mixer variant, the following is used:

```bash
python -m mixer.mixer_trl.train -c path/to/config.json
```

### 3. Evaluate

Depending on what is being evaluated, one of the following is typically used:

- `python -m mixer.hf_infer ...`
- `python -m lora.vllm_eval ...`
- `python -m classification.eval ...`
- `python -m mixed_eval.*`

## Dataset format

Most of the code expects JSONL files. The common task formats used here are:

- `math`: `question`, `answer`
- `boolq`: `question`, `passage`, `answer`
- `arc`: `question`, `choices`, `answerKey`
- `cola`: `sentence`, `label`
- `medqa`: `question`, `options`, `answer_idx`

The folder-level READMEs explain which scripts expect supervised prompt-completion data and which expect prompt-only RL-style data.

## Readme index

- [lora/README.md](./lora/README.md)
- [lora_offline/README.md](./lora_offline/README.md)
- [lora_trl/README.md](./lora_trl/README.md)
- [mixer/README.md](./mixer/README.md)
- [classification/README.md](./classification/README.md)
- [mixed_eval/README.md](./mixed_eval/README.md)
- [lorahub/README.md](./lorahub/README.md)
- [distill/README.md](./distill/README.md)
- [data/README.md](./data/README.md)
