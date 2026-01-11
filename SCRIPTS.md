# Scripts

```shell
python calibrate.py \
    --input aime_problems.json \
    --slm-model-id ~/huggingface/Qwen3-1.7B \
    --metric entropy \
    --percentiles 10,20,30 \
    --max-new-tokens 32768 \
    --output calibration_aime.json \
    --prompt-limit 15
```

```shell
python calibrate.py \
    --input gpqa_problems.json \
    --slm-model-id ~/huggingface/Qwen3-1.7B \
    --metric entropy \
    --percentiles 10,20,30 \
    --max-new-tokens 32768 \
    --output calibration_gpqa.json \
    --prompt-limit 30
```

- Details

```shell
python calibrate.py \
    --input aime_problems.json \
    --slm-model-id ~/huggingface/Qwen3-1.7B \
    --metric entropy \
    --percentiles 10,20,30 \
    --max-new-tokens 32768 \
    --output calibration_aime.json \
    --prompt-limit 1 \
    --details

```
