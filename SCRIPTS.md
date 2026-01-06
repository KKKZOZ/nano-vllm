# Scripts

```shell
python calibrate_aime.py \
    --slm-model-id ~/huggingface/Qwen3-1.7B \
    --metric entropy \
    --percentiles 10,20,30 \
    --max-new-tokens 32768 \
    --output calibration.json \
    --prompt-limit 1
```