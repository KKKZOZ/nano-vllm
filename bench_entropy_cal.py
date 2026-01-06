import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# === PyTorch 实现 ===
def _entropy_impl(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Core entropy calculation logic."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def calculate_token_entropy_torch(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    if temperature == 0.0:
        if logits.dim() == 1:
            return torch.tensor(0.0, device=logits.device)
        else:
            return torch.zeros(logits.shape[0], device=logits.device)
    return _entropy_impl(logits, temperature)


# === torch.compile 版本 ===
# Default compile
@torch.compile
def _entropy_impl_compile_default(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


# reduce-overhead mode - optimizes for reduced Python overhead
@torch.compile(mode="reduce-overhead")
def _entropy_impl_compile_reduce_overhead(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


# max-autotune mode - maximum autotuning for best performance
@torch.compile(mode="max-autotune")
def _entropy_impl_compile_max_autotune(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


# fullgraph=True - ensures no graph breaks
@torch.compile(fullgraph=True)
def _entropy_impl_compile_fullgraph(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


COMPILED_IMPLS = {
    "compile_default": _entropy_impl_compile_default,
    "compile_reduce_overhead": _entropy_impl_compile_reduce_overhead,
    "compile_max_autotune": _entropy_impl_compile_max_autotune,
    "compile_fullgraph": _entropy_impl_compile_fullgraph,
}


# === Triton 实现 ===
@triton.jit
def _token_entropy_kernel(
    logits_ptr,
    entropy_ptr,
    vocab_size,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx * vocab_size

    # Pass 1: Find max
    max_val = float("-inf")
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature
        max_val = tl.maximum(max_val, tl.max(scaled, axis=0))

    # Pass 2: Compute sum(exp)
    sum_exp = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature
        sum_exp += tl.sum(tl.exp(scaled - max_val), axis=0)

    log_sum_exp = tl.log(sum_exp)

    # Pass 3: Compute entropy
    entropy = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature

        shifted = scaled - max_val
        p = tl.exp(shifted) / sum_exp
        log_p = shifted - log_sum_exp

        entropy += tl.sum(tl.where(mask, -p * log_p, 0.0), axis=0)

    tl.store(entropy_ptr + row_idx, entropy)


def calculate_token_entropy_triton(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    squeeze_output = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True

    batch_size, vocab_size = logits.shape

    if temperature == 0.0:
        entropy = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
        return entropy.squeeze(0) if squeeze_output else entropy

    logits = logits.contiguous()
    entropy = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)

    BLOCK_SIZE = triton.next_power_of_2(min(vocab_size, 4096))

    _token_entropy_kernel[(batch_size,)](
        logits,
        entropy,
        vocab_size,
        temperature,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return entropy.squeeze(0) if squeeze_output else entropy


# === Benchmark ===
def benchmark():
    print("=" * 90)
    print(
        "Token Entropy Benchmark: PyTorch vs torch.compile vs Triton (Qwen3 vocab_size=151936)"
    )
    print("=" * 90)

    vocab_size = 151936  # Qwen3 vocab
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    temperature = 0.7
    num_runs = 5

    # All implementations to benchmark
    impl_names = [
        "torch",
        "compile_default",
        "compile_reduce_overhead",
        "compile_max_autotune",
        "compile_fullgraph",
        "triton",
    ]

    results = []

    # Global warmup - trigger compilation for all torch.compile variants
    print("\nWarming up and compiling...")
    for bs in [1, 32, 128]:
        warmup_logits = torch.randn(bs, vocab_size, device="cuda", dtype=torch.float32)
        for _ in range(10):
            _ = calculate_token_entropy_torch(warmup_logits, temperature)
            _ = calculate_token_entropy_triton(warmup_logits, temperature)
            for impl_fn in COMPILED_IMPLS.values():
                _ = impl_fn(warmup_logits, temperature)
        torch.cuda.synchronize()
    print("Warmup done.\n")

    for batch_size in batch_sizes:
        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

        row = {"batch_size": batch_size}

        # Benchmark each implementation
        for impl_name in impl_names:
            if impl_name == "torch":
                fn = lambda: calculate_token_entropy_torch(logits, temperature)
            elif impl_name == "triton":
                fn = lambda: calculate_token_entropy_triton(logits, temperature)
            else:
                impl_fn = COMPILED_IMPLS[impl_name]
                fn = lambda impl_fn=impl_fn: impl_fn(logits, temperature)

            times = []
            for _ in range(num_runs):
                ms = triton.testing.do_bench(fn, warmup=100, rep=500)
                times.append(ms)
            row[impl_name] = sum(times) / num_runs

        results.append(row)

    # Display name mapping
    display_names = {
        "torch": "torch",
        "compile_default": "compile",
        "compile_reduce_overhead": "reduce-oh",
        "compile_max_autotune": "max-auto",
        "compile_fullgraph": "fullgraph",
        "triton": "triton",
    }

    # Print table
    col_width = 10
    header = f"{'Batch':>6}"
    for name in impl_names:
        header += f" | {display_names[name]:>{col_width}}"

    print("Latency (ms):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['batch_size']:>6}"
        for name in impl_names:
            line += f" | {r[name]:>{col_width}.4f}"
        print(line)
    print("-" * len(header))

    # Print speedup table (relative to torch)
    print("\n" + "=" * 90)
    print("Speedup vs PyTorch (higher is better)")
    print("=" * 90)

    header = f"{'Batch':>6}"
    for name in impl_names[1:]:  # Skip torch itself
        header += f" | {display_names[name]:>{col_width}}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['batch_size']:>6}"
        torch_time = r["torch"]
        for name in impl_names[1:]:
            speedup = torch_time / r[name]
            line += f" | {speedup:>{col_width - 1}.2f}x"
        print(line)
    print("-" * len(header))


if __name__ == "__main__":
    benchmark()
