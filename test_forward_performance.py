import random
import time

import matplotlib.pyplot as plt
import torch

from nanovllm.backend import Backend


def test_forward_performance(model_name: str = "/root/huggingface/Qwen3-8B"):
    """
    Test the forward interface performance with different numbers of tokens.

    Steps:
    1. Initialize Backend
    2. Prefill with 256 tokens to reach KV cache length of 256
    3. Test decode phase with 1-256 tokens in a single forward call
    4. Plot the results
    """
    print(f"Initializing backend with model: {model_name}")
    backend = Backend(model_name, max_num_seqs=1)

    # Prepare initial tokens for prefill (256 tokens)
    seq_id = random.randint(0, 1000000)
    prefill_tokens = list(range(1, 257))  # Use token IDs 1-256 for prefill

    print(f"Prefilling with {len(prefill_tokens)} tokens...")
    start = time.perf_counter()
    logits = backend.forward(seq_id, prefill_tokens)
    prefill_time = time.perf_counter() - start
    print(f"Prefill time: {prefill_time:.4f}s")
    print(f"Logits shape: {logits.shape}")

    # Now test decode phase with different numbers of tokens
    # We'll test with: 1, 2, 4, 8, 16, 32, 64, 128, 256 tokens
    token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    times = []

    print("\nTesting decode phase with different token counts...")
    for num_tokens in token_counts:
        # Free the current sequence and start fresh
        # TODO: Figure out why this line causes issues
        # backend.free(seq_id)

        # # Re-prefill to get back to 256 tokens
        # backend.forward(seq_id, prefill_tokens)

        # # Now test decode with num_tokens
        decode_tokens = list(range(257, 257 + num_tokens))

        # # Warm up
        # backend.forward(seq_id, decode_tokens)

        # # Re-do the test for timing
        # backend.free(seq_id)
        backend.forward(seq_id, prefill_tokens)

        # Measure time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        logits = backend.forward(seq_id, decode_tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        print(
            f"  {num_tokens:3d} tokens: {elapsed * 1000:.2f}ms ({elapsed / num_tokens * 1000:.3f}ms per token)"
        )

    # Clean up
    backend.free(seq_id)
    backend.exit()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(
        token_counts, [t * 1000 for t in times], marker="o", linewidth=2, markersize=8
    )
    plt.xlabel("Number of Tokens", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.title(
        "Backend.forward() Performance: Decode Phase with Variable Token Counts\n(KV Cache Length = 256)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)

    # Add text annotations
    for i, (count, t) in enumerate(zip(token_counts, times)):
        plt.annotate(
            f"{t * 1000:.1f}ms",
            xy=(count, t * 1000),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig("forward_performance.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: forward_performance.png")

    # Also create a per-token time plot
    plt.figure(figsize=(10, 6))
    per_token_times = [t / count * 1000 for t, count in zip(times, token_counts)]
    plt.plot(
        token_counts,
        per_token_times,
        marker="s",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    plt.xlabel("Number of Tokens", fontsize=12)
    plt.ylabel("Time per Token (ms)", fontsize=12)
    plt.title(
        "Backend.forward() Per-Token Performance: Decode Phase\n(KV Cache Length = 256)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)

    # Add text annotations
    for i, (count, pt_time) in enumerate(zip(token_counts, per_token_times)):
        plt.annotate(
            f"{pt_time:.2f}ms",
            xy=(count, pt_time),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig("forward_performance_per_token.png", dpi=150, bbox_inches="tight")
    print("Per-token plot saved to: forward_performance_per_token.png")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Tokens':<10} {'Total Time (ms)':<20} {'Per Token (ms)':<20}")
    print("-" * 60)
    for count, t in zip(token_counts, times):
        print(f"{count:<10} {t * 1000:<20.2f} {t / count * 1000:<20.3f}")
    print("=" * 60)


if __name__ == "__main__":

    model = "/root/huggingface/Qwen3-8B"
    test_forward_performance(model)
