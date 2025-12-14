import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # Warmup: run multiple iterations to trigger torch.compile and CUDA optimizations
    print("Warming up...")
    warmup_rounds = 3
    for i in range(warmup_rounds):
        warmup_batch_size = min(32, num_seqs)
        warmup_prompt_ids = [
            [randint(0, 10000) for _ in range(randint(100, 512))]
            for _ in range(warmup_batch_size)
        ]
        warmup_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=128)
            for _ in range(warmup_batch_size)
        ]
        llm.generate(warmup_prompt_ids, warmup_params, use_tqdm=False)
        print(f"  Warmup round {i + 1}/{warmup_rounds} completed")
    print("Warmup completed. Starting benchmark...\n")

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
