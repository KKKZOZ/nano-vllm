"""
Example: Single-backend generation using solo strategy.

Demonstrates basic autoregressive generation using a single HuggingFace model
with the SoloStrategy. This is the simplest form of generation in the hybrid
generator framework.
"""

import argparse

import torch
from transformers import AutoTokenizer

from hybrid_generator.backends.hf import HFBackend
from hybrid_generator.backends.hybrid import BackendId, HybridBackend
from hybrid_generator.strategies.solo import SoloStrategy


def main():
    parser = argparse.ArgumentParser(description="Single-backend generation example")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/huggingface/Qwen3-1.7B",
        help="Model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for model (e.g., cuda:0)",
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
        "--backend",
        type=str,
        choices=["slm", "llm"],
        default="llm",
        help="Backend to use (slm or llm, both point to the same model)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--enable-stats-sync",
        action="store_true",
        help="Enable CUDA synchronization for accurate timing",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Solo Strategy - Single Backend Generation")
    print("=" * 80)
    print(f"Model: {args.model} on {args.device}")
    print(f"Backend ID: {args.backend}")
    print(f"Temperature: {args.temperature}")
    print("=" * 80)
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create backend
    print("Loading model...")
    backend = HFBackend(
        model_path=args.model,
        device=args.device,
        dtype=torch.float16,
    )

    # Create hybrid backend (both slm and llm point to the same backend)
    hybrid_backend = HybridBackend(slm_backend=backend, llm_backend=backend)

    # Create strategy
    strategy = SoloStrategy()

    # Convert backend string to BackendId enum
    backend_id = BackendId.SLM if args.backend == "slm" else BackendId.LLM

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
        device=args.device,
        verbose=args.verbose,
        enable_stats_sync=args.enable_stats_sync,
        backend_id=backend_id,
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
    print(f"Backend used: {stats['backend']}")
    print(f"Decode steps: {stats['decode_steps']}")
    print(f"Elapsed time: {stats['elapsed_time']:.2f}s")
    print(f"Throughput: {stats['total_tokens'] / stats['elapsed_time']:.2f} tokens/s")
    print()

    # Cleanup
    hybrid_backend.exit()


if __name__ == "__main__":
    main()
