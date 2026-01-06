
## AIME 24

```shell
set -x CUDA_VISIBLE_DEVICES 0
set -x MASTER_PORT 2333
python eval_hybrid.py \
    --slm /root/huggingface/Qwen3-1.7B \
    --llm /root/huggingface/Qwen3-8B \
    --strategy entropy \
    --threshold 0.1 \
    --report-routing-metrics \
    --apply-chat-template \
    --tasks aime24 \
    --gen_kwargs '{"max_gen_toks": 32768}' \
    --limit 30
```

```shell
set -x CUDA_VISIBLE_DEVICES 2
set -x MASTER_PORT 2335
python eval_hybrid.py \
    --slm /root/huggingface/Qwen3-0.6B \
    --llm /root/huggingface/Qwen3-8B \
    --strategy entropy \
    --threshold 0.1 \
    --report-routing-metrics \
    --apply-chat-template \
    --tasks aime24 \
    --gen_kwargs '{"max_gen_toks": 32768}' \
    --limit 30
```

## AIME 25

```shell
set -x CUDA_VISIBLE_DEVICES 1
set -x MASTER_PORT 2334
python eval_hybrid.py \
    --slm /root/huggingface/Qwen3-1.7B \
    --llm /root/huggingface/Qwen3-8B \
    --strategy entropy \
    --threshold 0.5 \
    --report-routing-metrics \
    --apply-chat-template \
    --tasks aime25 \
    --gen_kwargs '{"max_gen_toks": 32768}' \
    --limit 30
```

```shell
set -x CUDA_VISIBLE_DEVICES 3
set -x MASTER_PORT 2336
python eval_hybrid.py \
    --slm /root/huggingface/Qwen3-0.6B \
    --llm /root/huggingface/Qwen3-8B \
    --strategy entropy \
    --threshold 0.1 \
    --report-routing-metrics \
    --apply-chat-template \
    --tasks aime25 \
    --gen_kwargs '{"max_gen_toks": 32768}' \
    --limit 30
```

## GPAQ

```shell
set -x CUDA_VISIBLE_DEVICES 2
set -x MASTER_PORT 2333
python eval_hybrid.py \
    --slm /root/huggingface/Qwen3-1.7B \
    --llm /root/huggingface/Qwen3-8B \
    --strategy entropy \
    --threshold 0.1 \
    --report-routing-metrics \
    --apply-chat-template \
    --tasks gpqa_diamond_cot_n_shot \
    --gen_kwargs '{"max_gen_toks": 32768}' \
    --limit 200
```
```shell
set -x CUDA_VISIBLE_DEVICES 3
set -x MASTER_PORT 2334
python eval_hybrid.py \
    --slm /root/huggingface/Qwen3-0.6B \
    --llm /root/huggingface/Qwen3-8B \
    --strategy entropy \
    --threshold 0.1 \
    --report-routing-metrics \
    --apply-chat-template \
    --tasks gpqa_diamond_cot_n_shot \
    --gen_kwargs '{"max_gen_toks": 32768}' \
    --limit 200
```

```shell
set -x VLLM_WORKER_MULTIPROC_METHOD spawn
set -x CUDA_VISIBLE_DEVICES 2,3
lm-eval --model vllm \
    --model_args pretrained=/root/huggingface/Qwen3-8B,tensor_parallel_size=2 \
    --tasks gpqa_diamond_cot_n_shot \
    --apply_chat_template \
    --batch_size 32 \
    --limit 200 \
    --gen_kwargs max_gen_toks=32768,temperature=0.6,top_p=0.95,top_k=20,min_p=0,do_sample=True \
    --log_samples \
    --output_path ./results/gpqa
```
```