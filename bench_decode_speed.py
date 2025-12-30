import argparse
import os
from statistics import mean
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.backend import Backend
from nanovllm.layers.sampler import Sampler


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def require_model_path(path: str) -> str:
    if not path:
        raise SystemExit("Missing model path. Use --model or set NANOVLLM_MODEL.")
    if not os.path.isdir(path):
        raise SystemExit(f"Model path not found: {path}")
    return path


def build_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def measure_llm_decode(
    llm: LLM, prompt_ids: list[int], sampling_params: SamplingParams
):
    llm.add_request(prompt_ids, sampling_params)
    decode_tokens = 0
    decode_time = 0.0
    while not llm.is_finished():
        cuda_sync()
        t0 = perf_counter()
        _, num_tokens = llm.step()
        cuda_sync()
        step_time = perf_counter() - t0
        if num_tokens < 0:
            decode_tokens += -num_tokens
            decode_time += step_time
    return decode_tokens, decode_time


def measure_backend_decode(
    backend: Backend,
    sampler: Sampler,
    temperature: torch.Tensor,
    prompt_ids: list[int],
    max_tokens: int,
    ignore_eos: bool,
    seq_id: int,
):
    logits = backend.forward(seq_id, prompt_ids)
    first_logits = logits[-1].unsqueeze(0)
    token_id = sampler(first_logits, temperature).item()

    decode_tokens = 0
    decode_time = 0.0
    target_decode_tokens = max(0, max_tokens - 1)

    for _ in range(target_decode_tokens):
        cuda_sync()
        t0 = perf_counter()
        logits = backend.forward(seq_id, [token_id])
        token_id = sampler(logits[-1].unsqueeze(0), temperature).item()
        cuda_sync()
        step_time = perf_counter() - t0
        decode_tokens += 1
        decode_time += step_time
        if not ignore_eos and token_id == backend.eos:
            break

    backend.free(seq_id)
    return decode_tokens, decode_time


def format_runs(name: str, runs: list[tuple[int, float]]):
    speeds = []
    print(f"{name}:")
    for i, (tokens, seconds) in enumerate(runs, start=1):
        speed = tokens / max(seconds, 1e-9)
        speeds.append(speed)
        print(
            f"  run {i}: {tokens} decode tokens, {seconds * 1000:.2f} ms, "
            f"{speed:.2f} tok/s"
        )
    print(f"  mean: {mean(speeds):.2f} tok/s")
    return speeds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare decode speed between LLMEngine and Backend for the same model."
        )
    )
    parser.add_argument("--model", default="/root/huggingface/Qwen3-1.7B/")
    parser.add_argument("--prompt", default="introduce yourself")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument("--ignore-eos", action="store_true", default=True)
    eos_group.add_argument("--respect-eos", action="store_true")

    args = parser.parse_args()
    model_path = require_model_path(args.model)
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be >= 2 to measure decode speed.")

    ignore_eos = args.ignore_eos and not args.respect_eos

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    llm = LLM(
        model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
    )
    prompt_text = build_prompt(llm.tokenizer, args.prompt)
    prompt_ids = llm.tokenizer.encode(prompt_text)

    print("Config:")
    print(f"  model: {model_path}")
    print(f"  prompt tokens: {len(prompt_ids)}")
    print(f"  completion tokens: {args.max_tokens}")
    print(f"  decode steps: {max(0, args.max_tokens - 1)}")
    print(f"  ignore eos: {ignore_eos}")
    print(f"  runs: {args.runs}")
    print("")

    if args.warmup_tokens > 0:
        warmup_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.warmup_tokens,
            ignore_eos=True,
        )
        measure_llm_decode(llm, prompt_ids, warmup_params)

    llm_runs = []
    for _ in range(args.runs):
        sp = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            ignore_eos=ignore_eos,
        )
        llm_runs.append(measure_llm_decode(llm, prompt_ids, sp))

    llm.exit()
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    backend = Backend(
        model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
    )
    sampler = Sampler()
    temperature = torch.tensor(
        [args.temperature], dtype=torch.float32, pin_memory=True
    ).cuda(non_blocking=True)

    if args.warmup_tokens > 0:
        measure_backend_decode(
            backend,
            sampler,
            temperature,
            prompt_ids,
            args.warmup_tokens,
            True,
            10_000,
        )

    backend_runs = []
    seq_id = 20_000
    for _ in range(args.runs):
        backend_runs.append(
            measure_backend_decode(
                backend,
                sampler,
                temperature,
                prompt_ids,
                args.max_tokens,
                ignore_eos,
                seq_id,
            )
        )
        seq_id += 1

    backend.exit()

    format_runs("LLMEngine decode", llm_runs)
    print("")
    format_runs("Backend decode", backend_runs)


if __name__ == "__main__":
    main()
