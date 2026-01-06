import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 1. Triton Kernel: Looped Implementation (修复 NaN 版)
# -----------------------------------------------------------------------------


@triton.jit
def _entropy_kernel_looped(
    logits_ptr,
    output_ptr,
    stride_logits_row,
    stride_logits_col,
    temperature,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = logits_ptr + row_idx * stride_logits_row

    # 初始化 Accumulators (显式使用 float32)
    m_global = -float("inf")
    s_global = 0.0
    z_global = 0.0

    # 循环遍历词表
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # 1. Load Chunk
        a_ptr = row_start_ptr + cols * stride_logits_col
        val = tl.load(a_ptr, mask=mask, other=-float("inf"))
        val = val.to(tl.float32)  # [Fix] 保证精度
        val = val / temperature

        # 2. Compute Chunk Stats
        m_local = tl.max(val, axis=0)

        # Safe masking
        val_safe = tl.where(mask, val, 0.0)
        p_exp_local = tl.exp(val - m_local)
        s_local = tl.sum(p_exp_local, axis=0)
        w_exp_local = (val_safe - m_local) * p_exp_local
        z_local = tl.sum(w_exp_local, axis=0)

        # 3. Online Update
        # [Crucial Fix 1] 使用 tl.maximum 避免歧义
        m_new = tl.maximum(m_global, m_local)

        # [Crucial Fix 2] 处理初始化时的 -inf * 0 = NaN 问题
        # 如果 s_global 为 0 (说明是第一次迭代或之前全是 padding)，
        # 我们让 m_global_safe = m_new，这样 diff 就为 0，避免出现 -inf
        m_global_safe = tl.where(s_global == 0, m_new, m_global)

        alpha = tl.exp(m_global_safe - m_new)
        beta = tl.exp(m_local - m_new)

        # 更新 S 和 Z
        s_new = s_global * alpha + s_local * beta

        # 更新 Z (分子部分)
        # 注意这里使用 m_global_safe 参与计算
        z_global = alpha * (z_global + (m_global_safe - m_new) * s_global) + beta * (
            z_local + (m_local - m_new) * s_local
        )

        m_global = m_new
        s_global = s_new

    # 4. Final Entropy Calculation
    # 防止 s_global 为 0 (例如全是 padding) 导致除零
    entropy = tl.log(s_global) - (z_global / s_global)
    tl.store(output_ptr + row_idx, entropy)


def entropy_triton_qwen(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    n_rows, n_cols = logits.shape
    entropy = torch.empty(n_rows, dtype=logits.dtype, device=logits.device)

    BLOCK_SIZE = 4096
    num_warps = 8

    grid = (n_rows,)
    _entropy_kernel_looped[grid](
        logits,
        entropy,
        logits.stride(0),
        logits.stride(1),
        temperature,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return entropy


# -----------------------------------------------------------------------------
# 2. PyTorch Reference
# -----------------------------------------------------------------------------


def entropy_pytorch(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    # PyTorch Eager reference
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


# -----------------------------------------------------------------------------
# 3. Benchmark Script
# -----------------------------------------------------------------------------


def run_qwen_benchmark():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")

    QWEN_VOCAB_SIZE = 151936
    TEMP = 0.7
    batch_sizes = [1, 2, 4, 8, 16, 32]

    results = []

    print(f"Running Benchmark for Qwen-3 (Vocab: {QWEN_VOCAB_SIZE})...")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    for bs in batch_sizes:
        x = torch.randn(bs, QWEN_VOCAB_SIZE, device=device, dtype=torch.float16)

        # 1. Warmup & Correctness Check
        try:
            y_ref = entropy_pytorch(x, TEMP)
            y_tri = entropy_triton_qwen(x, TEMP)

            # 使用 nan_to_num 处理可能的 nan 以便输出具体的 diff 报错
            diff = torch.abs(y_ref - y_tri)
            max_diff = torch.max(torch.nan_to_num(diff, nan=9999.0)).item()

            if not torch.allclose(y_ref, y_tri, atol=1e-3, rtol=1e-3):
                print(f"[Warning] Batch {bs}: Mismatch! Max diff: {max_diff:.6f}")
        except Exception as e:
            print(f"[Error] Batch {bs}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # 2. Measure Latency
        # [Fix] 显式传递 quantiles 以确保返回 tuple (median, p20, p80)
        # 注意: do_bench 返回的是毫秒 (ms)
        ms_torch, min_ms_torch, max_ms_torch = triton.testing.do_bench(
            lambda: entropy_pytorch(x, TEMP),
            warmup=25,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )

        ms_triton, min_ms_triton, max_ms_triton = triton.testing.do_bench(
            lambda: entropy_triton_qwen(x, TEMP),
            warmup=25,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )

        # 3. Metrics
        us_torch = ms_torch * 1000
        us_triton = ms_triton * 1000
        us_triton_min = min_ms_triton * 1000
        us_triton_max = max_ms_triton * 1000
        speedup = ms_torch / ms_triton

        total_bytes = bs * QWEN_VOCAB_SIZE * 2
        gbps = (total_bytes * 1e-9) / (ms_triton * 1e-3)

        results.append(
            {
                "Batch": bs,
                "Torch (us)": round(us_torch, 1),
                "Triton Avg (us)": round(us_triton, 1),
                "Triton P20 (us)": round(us_triton_min, 1),  # 使用 P20 代替 Min
                "Triton P80 (us)": round(us_triton_max, 1),  # 使用 P80 代替 Max
                "Speedup": round(speedup, 2),
                "Bandwidth (GB/s)": round(gbps, 1),
            }
        )

    df = pd.DataFrame(results)
    print("\nBenchmark Results (Latency in Microseconds):")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))


if __name__ == "__main__":
    run_qwen_benchmark()
