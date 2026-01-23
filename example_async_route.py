"""
Example: Async entropy-based routing strategy.

Demonstrates concurrent execution of SLM and LLM on separate GPUs.
When SLM detects high entropy, it triggers LLM asynchronously and continues
generating speculatively.
"""

import argparse

import torch
from transformers import AutoTokenizer

from hybrid_generator.backends.hf import HFBackend
from hybrid_generator.backends.hybrid import AsyncHybridBackend
from hybrid_generator.strategies.async_route import AsyncEntropyStrategy


def main():
    parser = argparse.ArgumentParser(description="Async entropy-based routing example")
    parser.add_argument(
        "--slm-model",
        type=str,
        default="/root/huggingface/Qwen3-1.7B",
        help="Small language model path",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="/root/huggingface/Qwen3-8B",
        help="Large language model path",
    )
    parser.add_argument(
        "--slm-device",
        type=str,
        default="cuda:0",
        help="Device for SLM (e.g., cuda:0)",
    )
    parser.add_argument(
        "--llm-device",
        type=str,
        default="cuda:1",
        help="Device for LLM (e.g., cuda:1)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms:",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum new tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Entropy threshold for routing (higher = more LLM usage)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--report-metrics", action="store_true", help="Report live metrics"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Async Entropy-Based Routing Strategy")
    print("=" * 80)
    print(f"SLM: {args.slm_model} on {args.slm_device}")
    print(f"LLM: {args.llm_model} on {args.llm_device}")
    print(f"Entropy threshold: {args.threshold}")
    print(f"Temperature: {args.temperature}")
    print("=" * 80)
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.slm_model)

    # Create backends
    print("Loading models...")
    slm_backend = HFBackend(
        model_path=args.slm_model,
        device=args.slm_device,
        dtype=torch.float16,
    )
    llm_backend = HFBackend(
        model_path=args.llm_model,
        device=args.llm_device,
        dtype=torch.float16,
    )

    # Create async hybrid backend
    hybrid_backend = AsyncHybridBackend(slm_backend, llm_backend)

    # Create strategy
    strategy = AsyncEntropyStrategy()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Generating...\n")
    print(f"Prompt: {input}\n")

    # Generate
    output, stats = strategy.generate(
        hybrid_backend=hybrid_backend,
        tokenizer=tokenizer,
        prompt=input,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=0.0,
        device=args.slm_device,
        threshold=args.threshold,
        verbose=args.verbose,
        report_live_metrics=args.report_metrics,
    )

    print("\n")
    print("=" * 80)
    print("Generation Complete")
    print("=" * 80)

    if not args.verbose:
        print(f"\nOutput:\n{output}\n")

    print("\nStatistics:")
    print("-" * 80)
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"SLM tokens: {stats['slm_tokens']}")
    print(f"LLM tokens: {stats['llm_tokens']}")
    print(f"Decode steps: {stats['decode_steps']}")
    print(f"Elapsed time: {stats['elapsed_time']:.2f}s")
    print(f"Throughput: {stats['total_tokens'] / stats['elapsed_time']:.2f} tokens/s")
    print()

    print("Async Routing Metrics:")
    print("-" * 80)
    print(f"Speculation attempts: {stats['speculation_attempts']}")
    print(f"Speculation accepted: {stats['speculation_accepted']}")
    print(f"Speculation rejected: {stats['speculation_rejected']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Avg speculation length: {stats['avg_speculation_length']:.2f} tokens")
    print(
        f"Avg SLM cumulative prob (accepted): {stats['avg_slm_cumprob_accepted']:.4f}"
    )
    print(
        f"Avg LLM cumulative prob (accepted): {stats['avg_llm_cumprob_accepted']:.4f}"
    )
    print()

    print("Model Usage:")
    print("-" * 80)
    slm_pct = (
        stats["slm_tokens"] / stats["total_tokens"] * 100
        if stats["total_tokens"] > 0
        else 0
    )
    llm_pct = (
        stats["llm_tokens"] / stats["total_tokens"] * 100
        if stats["total_tokens"] > 0
        else 0
    )
    print(f"SLM: {slm_pct:.1f}% of tokens")
    print(f"LLM: {llm_pct:.1f}% of tokens")

    # Cleanup
    hybrid_backend.exit()


if __name__ == "__main__":
    main()
