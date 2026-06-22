# `mixed_eval`

This folder contains the mixed-task evaluation code used here, mainly for GSM8K + BoolQ.

The idea is:

1. build a mixed prompt dataset
2. run prediction with either the mixer or routed adapters
3. measure task-wise and joint accuracy

## Build the mixed prompt file

For random prompt order:

```bash
python -m mixed_eval.dataset \
  --math-path data/gsm8k/test.jsonl \
  --boolq-path data/boolq/validation.jsonl \
  --output outputs/mixed_random_order.jsonl \
  --seed 13
```

For a fixed `math_first` order:

```bash
python -m mixed_eval.dataset \
  --math-path data/gsm8k/test.jsonl \
  --boolq-path data/boolq/validation.jsonl \
  --output outputs/mixed_math_first.jsonl \
  --seed 13 \
  --prompt-order math_first
```

## Predict with the mixer

```bash
python -m mixed_eval.predict \
  -c mixer_config.json \
  --dataset-name mixed_gsm8k_boolq \
  --dataset-path outputs/mixed_random_order.jsonl \
  --output outputs/mixed_random_order_preds.jsonl
```

## Evaluate the predictions

```bash
python -m mixed_eval.evaluate \
  --predictions outputs/mixed_random_order_preds.jsonl \
  --scored-output outputs/mixed_random_order_scored.jsonl \
  --summary-output outputs/mixed_random_order_summary.json
```

If weighted evaluation is needed because of oversampling, the following is used:

```bash
python -m mixed_eval.evaluate_weighted \
  --predictions outputs/mixed_random_order_preds.jsonl \
  --summary-output outputs/mixed_random_order_weighted_summary.json
```

## Classification routing baseline

First, classifier inputs are created from the mixed dataset:

```bash
python -m mixed_eval.classification_bridge \
  --mixed-path outputs/mixed_random_order.jsonl \
  --output-dir outputs/classification_bridge \
  --base-classification-config classification/config.json \
  --order-mode from_mixed
```

Then the classifier is run:

```bash
python -m classification.eval \
  -c outputs/classification_bridge/classification_config.json \
  --checkpoint-dir outputs/bert_cls/final_checkpoint \
  --output-dir outputs/classification_router \
  --use-validation
```

Then the classifier outputs are mapped back to the mixed prompts:

```bash
python -m mixed_eval.classification_route_map \
  --mixed-path outputs/mixed_random_order.jsonl \
  --classification-manifest outputs/classification_bridge/classification_manifest.jsonl \
  --classification-predictions outputs/classification_router/gsm8k.jsonl \
  --output outputs/routing_manifest.jsonl
```

## Routed vLLM prediction

If a routing manifest with adapter paths is already available, the following is used:

```bash
python -m mixed_eval.vllm_predict \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --routing-manifest outputs/routing_manifest.jsonl \
  --output outputs/mixed_vllm_preds.jsonl \
  --max-new-tokens 1280 \
  --num-samples 1 \
  --temperature 0 \
  --max-lora-rank 128
```

## Note

This folder is only for evaluation. It does not train models by itself.
