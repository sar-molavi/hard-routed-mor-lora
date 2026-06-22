# Classification Summary

## Accuracy

| Dataset | Total | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| cola | 1043 | 1037 | 99.42% |
| arc | 1172 | 1159 | 98.89% |
| medqa | 1273 | 1268 | 99.61% |
| gsm8k | 1319 | 1319 | 100.00% |
| boolq | 3270 | 3270 | 100.00% |
| Overall | 8077 | 8053 | 99.70% |

## Training Config

| Field | Value |
| --- | --- |
| Model | `bert-base-uncased` |
| Output dir | `./classification/bert_cls` |
| Num train epochs | `3` |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.06` |
| Train batch size | `8` |
| Eval batch size | `8` |
| Gradient accumulation steps | `1` |
| Gradient checkpointing | `false` |
| Save steps | `100` |
| Eval steps | `100` |
| Save total limit | `2` |
| Logging steps | `100` |
| Max length | `256` |
| Max grad norm | `1.0` |
| BF16 | `false` |
| Early stopping patience | `2` |
| Max train samples | `1000` |
| Save strategy | `steps` |
| Load best model at end | `true` |
| Eval strategy | `steps` |
| LR scheduler type | `cosine` |
| Seed | `42` |
| Report to | `tensorboard` |

## Datasets

| Dataset | Path |
| --- | --- |
| cola | `data/cola/validation.jsonl` |
| arc | `data/arc-c/test.jsonl` |
| medqa | `data/medqa/test.jsonl` |
| gsm8k | `data/gsm8k/test.jsonl` |
| boolq | `data/boolq/validation.jsonl` |
