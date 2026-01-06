import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# === PyTorch 基准实现 ===
def calculate_token_entropy_torch(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    if temperature == 0.0:
        if logits.dim() == 1:
            return torch.tensor(0.0, device=logits.device)
        else:
            return torch.zeros(logits.shape[0], device=logits.device)

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


# === V1: 原始 Triton 实现 ===
@triton.jit
def _token_entropy_kernel_v1(
    logits_ptr,
    entropy_ptr,
    vocab_size,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx * vocab_size

    max_val = float("-inf")
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature
        max_val = tl.maximum(max_val, tl.max(scaled, axis=0))

    sum_exp = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature
        sum_exp += tl.sum(tl.exp(scaled - max_val), axis=0)

    log_sum_exp = tl.log(sum_exp)

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


def calculate_token_entropy_triton_v1(
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

    _token_entropy_kernel_v1[(batch_size,)](
        logits,
        entropy,
        vocab_size,
        temperature,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return entropy.squeeze(0) if squeeze_output else entropy


# === V2: 减少 Python 开销 ===
@triton.jit
def _token_entropy_kernel_v2(
    logits_ptr,
    entropy_ptr,
    vocab_size,
    inv_temp,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx * vocab_size

    max_val = float("-inf")
    for off in range(0, vocab_size, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_start + offs, mask=mask, other=float("-inf"))
        x = x * inv_temp
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    sum_exp = 0.0
    for off in range(0, vocab_size, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_start + offs, mask=mask, other=float("-inf"))
        x = x * inv_temp - max_val
        sum_exp += tl.sum(tl.where(mask, tl.exp(x), 0.0), axis=0)

    log_sum_exp = tl.log(sum_exp)

    entropy = 0.0
    for off in range(0, vocab_size, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_start + offs, mask=mask, other=float("-inf"))
        x = x * inv_temp - max_val
        p = tl.exp(x) / sum_exp
        log_p = x - log_sum_exp
        entropy += tl.sum(tl.where(mask, -p * log_p, 0.0), axis=0)

    tl.store(entropy_ptr + row_idx, entropy)


def calculate_token_entropy_triton_v2(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    orig_dim = logits.dim()
    if orig_dim == 1:
        logits = logits.unsqueeze(0)

    batch_size, vocab_size = logits.shape
    entropy = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)

    _token_entropy_kernel_v2[(batch_size,)](
        logits,
        entropy,
        vocab_size,
        1.0 / temperature,
        BLOCK_SIZE=4096,
    )

    return entropy[0] if orig_dim == 1 else entropy


# === V3: CUDA Graph 版本 ===
class TokenEntropyGraphed:
    def __init__(
        self, batch_size: int, vocab_size: int, temperature: float, device="cuda"
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.inv_temp = 1.0 / temperature

        self.logits_buffer = torch.empty(
            batch_size, vocab_size, device=device, dtype=torch.float32
        )
        self.entropy_buffer = torch.empty(
            batch_size, device=device, dtype=torch.float32
        )

        # Warmup
        for _ in range(3):
            _token_entropy_kernel_v2[(batch_size,)](
                self.logits_buffer,
                self.entropy_buffer,
                vocab_size,
                self.inv_temp,
                BLOCK_SIZE=4096,
            )
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            _token_entropy_kernel_v2[(batch_size,)](
                self.logits_buffer,
                self.entropy_buffer,
                vocab_size,
                self.inv_temp,
                BLOCK_SIZE=4096,
            )

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.logits_buffer.copy_(logits)
        self.graph.replay()
        return self.entropy_buffer


# === Benchmark ===
def benchmark():
    print("=" * 70)
    print("Token Entropy Benchmark (Qwen3 focused)")
    print("=" * 70)

    # Qwen3 实际配置
    QWEN3_VOCAB_SIZE = 151936

    configs = [
        # (batch_size, vocab_size, description)
        (1, QWEN3_VOCAB_SIZE, "Qwen3 single"),
        (4, QWEN3_VOCAB_SIZE, "Qwen3 batch=4"),
        (8, QWEN3_VOCAB_SIZE, "Qwen3 batch=8"),
        (16, QWEN3_VOCAB_SIZE, "Qwen3 batch=16"),
        (32, QWEN3_VOCAB_SIZE, "Qwen3 batch=32"),
    ]

    temperature = 0.7
    warmup = 100
    rep = 500

    for batch_size, vocab_size, desc in configs:
        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

        entropy_graphed = TokenEntropyGraphed(batch_size, vocab_size, temperature)

        # Warmup
        for _ in range(10):
            _ = calculate_token_entropy_torch(logits, temperature)
            _ = calculate_token_entropy_triton_v1(logits, temperature)
            _ = calculate_token_entropy_triton_v2(logits, temperature)
            _ = entropy_graphed(logits)
        torch.cuda.synchronize()

        # Correctness
        out_torch = calculate_token_entropy_torch(logits, temperature)
        out_v1 = calculate_token_entropy_triton_v1(logits, temperature)
        out_v2 = calculate_token_entropy_triton_v2(logits, temperature)
        out_graph = entropy_graphed(logits)

        diff_v1 = (out_torch - out_v1).abs().max().item()
        diff_v2 = (out_torch - out_v2).abs().max().item()
        diff_graph = (out_torch - out_graph).abs().max().item()

        # Benchmark
        ms_torch = triton.testing.do_bench(
            lambda: calculate_token_entropy_torch(logits, temperature),
            warmup=warmup,
            rep=rep,
        )
        ms_v1 = triton.testing.do_bench(
            lambda: calculate_token_entropy_triton_v1(logits, temperature),
            warmup=warmup,
            rep=rep,
        )
        ms_v2 = triton.testing.do_bench(
            lambda: calculate_token_entropy_triton_v2(logits, temperature),
            warmup=warmup,
            rep=rep,
        )
        ms_graph = triton.testing.do_bench(
            lambda: entropy_graphed(logits),
            warmup=warmup,
            rep=rep,
        )

        print(f"\n{desc}: ({batch_size}, {vocab_size})")
        print(f"  {'Method':<15} {'Time (ms)':<12} {'vs PyTorch':<12} {'Max Diff':<12}")
        print(f"  {'-' * 51}")
        print(f"  {'PyTorch':<15} {ms_torch:<12.4f} {'1.00x':<12} {'-':<12}")
        print(
            f"  {'Triton V1':<15} {ms_v1:<12.4f} {ms_torch / ms_v1:<12.2f}x {diff_v1:<12.2e}"
        )
        print(
            f"  {'Triton V2':<15} {ms_v2:<12.4f} {ms_torch / ms_v2:<12.2f}x {diff_v2:<12.2e}"
        )
        print(
            f"  {'CUDA Graph':<15} {ms_graph:<12.4f} {ms_torch / ms_graph:<12.2f}x {diff_graph:<12.2e}"
        )

    # 额外测试：模拟推理场景下的连续调用
    print("\n" + "=" * 70)
    print("Latency Test: 1000 consecutive calls (batch=1, simulating inference)")
    print("=" * 70)

    batch_size, vocab_size = 1, QWEN3_VOCAB_SIZE
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
    entropy_graphed = TokenEntropyGraphed(batch_size, vocab_size, temperature)

    import time

    n_calls = 1000

    # PyTorch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = calculate_token_entropy_torch(logits, temperature)
    torch.cuda.synchronize()
    pytorch_total = (time.perf_counter() - t0) * 1000

    # V1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = calculate_token_entropy_triton_v1(logits, temperature)
    torch.cuda.synchronize()
    v1_total = (time.perf_counter() - t0) * 1000

    # V2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = calculate_token_entropy_triton_v2(logits, temperature)
    torch.cuda.synchronize()
    v2_total = (time.perf_counter() - t0) * 1000

    # CUDA Graph
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = entropy_graphed(logits)
    torch.cuda.synchronize()
    graph_total = (time.perf_counter() - t0) * 1000

    print(
        f"\n  {'Method':<15} {'Total (ms)':<15} {'Per call (us)':<15} {'vs PyTorch':<12}"
    )
    print(f"  {'-' * 57}")
    print(
        f"  {'PyTorch':<15} {pytorch_total:<15.2f} {pytorch_total / n_calls * 1000:<15.2f} {'1.00x':<12}"
    )
    print(
        f"  {'Triton V1':<15} {v1_total:<15.2f} {v1_total / n_calls * 1000:<15.2f} {pytorch_total / v1_total:<12.2f}x"
    )
    print(
        f"  {'Triton V2':<15} {v2_total:<15.2f} {v2_total / n_calls * 1000:<15.2f} {pytorch_total / v2_total:<12.2f}x"
    )
    print(
        f"  {'CUDA Graph':<15} {graph_total:<15.2f} {graph_total / n_calls * 1000:<15.2f} {pytorch_total / graph_total:<12.2f}x"
    )


if __name__ == "__main__":
    benchmark()
