# benchmark_entropy_triton_qwen3.py
# ------------------------------------------------------------
# Benchmark entropy = -sum softmax(x/T) * log_softmax(x/T)
# Optimized identity: H = logZ - E_p[x], x = logits/T, logZ = log(sum exp(x))
# Qwen3 vocab_size default: 151936  (model config default)
#
# Run:
#   python benchmark_entropy_triton_qwen3.py
#   python benchmark_entropy_triton_qwen3.py --dtype bf16 --temperature 0.7
#   python benchmark_entropy_triton_qwen3.py --B 1 2 4 --S 1 16 32 --repeats 200
# ------------------------------------------------------------

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------- Triton Kernels -------------------------


@triton.jit
def _partial_max_kernel(
    X_ptr,
    PM_ptr,  # partial max
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_pmm: tl.constexpr,
    stride_pmb: tl.constexpr,
    n_cols,
    inv_temp,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    pid_b = tl.program_id(1)  # block id along vocab
    row_ptr = X_ptr + pid_m * stride_xm

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(row_ptr + offs * stride_xn, mask=mask, other=-float("inf"))
    x = x.to(tl.float32) * inv_temp

    m = tl.max(x, axis=0)
    tl.store(PM_ptr + pid_m * stride_pmm + pid_b * stride_pmb, m)


@triton.jit
def _reduce_max_kernel(
    PM_ptr,
    M_ptr,  # partial max -> max
    stride_pmm: tl.constexpr,
    stride_pmb: tl.constexpr,
    n_blocks,
    REDUCE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, REDUCE_BLOCK)
    mask = offs < n_blocks

    pm = tl.load(
        PM_ptr + pid_m * stride_pmm + offs * stride_pmb, mask=mask, other=-float("inf")
    )
    m = tl.max(pm.to(tl.float32), axis=0)
    tl.store(M_ptr + pid_m, m)


@triton.jit
def _partial_sums_kernel(
    X_ptr,
    M_ptr,
    PS_ptr,
    PSX_ptr,  # max + partial sums
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_psm: tl.constexpr,
    stride_psb: tl.constexpr,
    n_cols,
    inv_temp,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    row_ptr = X_ptr + pid_m * stride_xm

    maxv = tl.load(M_ptr + pid_m).to(tl.float32)

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(row_ptr + offs * stride_xn, mask=mask, other=0.0)
    x = x.to(tl.float32) * inv_temp

    e = tl.exp(x - maxv)
    e = tl.where(mask, e, 0.0)

    s = tl.sum(e, axis=0)
    sx = tl.sum(e * x, axis=0)

    tl.store(PS_ptr + pid_m * stride_psm + pid_b * stride_psb, s)
    tl.store(PSX_ptr + pid_m * stride_psm + pid_b * stride_psb, sx)


@triton.jit
def _reduce_sums_kernel(
    M_ptr,
    PS_ptr,
    PSX_ptr,
    Out_ptr,
    stride_psm: tl.constexpr,
    stride_psb: tl.constexpr,
    n_blocks,
    REDUCE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    maxv = tl.load(M_ptr + pid_m).to(tl.float32)

    offs = tl.arange(0, REDUCE_BLOCK)
    mask = offs < n_blocks

    ps = tl.load(
        PS_ptr + pid_m * stride_psm + offs * stride_psb, mask=mask, other=0.0
    ).to(tl.float32)
    psx = tl.load(
        PSX_ptr + pid_m * stride_psm + offs * stride_psb, mask=mask, other=0.0
    ).to(tl.float32)

    sum_exp = tl.sum(ps, axis=0)
    sum_exp_x = tl.sum(psx, axis=0)

    logZ = tl.log(sum_exp) + maxv
    entropy = logZ - (sum_exp_x / sum_exp)

    tl.store(Out_ptr + pid_m, entropy)


# ------------------------- Triton Wrapper -------------------------


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def entropy_triton_qwen3(
    logits: torch.Tensor,
    temperature: float,
    block: int = 2048,
    num_warps: int = 8,
) -> torch.Tensor:
    """
    Entropy over last dim using 4-kernel Triton reduction.
    Returns fp32 tensor of shape logits.shape[:-1].
    """
    assert logits.is_cuda, "logits must be CUDA tensor"
    assert temperature > 0.0
    assert logits.ndim >= 2, "expect [..., vocab]"
    assert logits.dtype in (torch.float16, torch.bfloat16, torch.float32)

    n_cols = logits.shape[-1]
    x2d = logits.reshape(-1, n_cols)  # [M, V]
    M = x2d.shape[0]

    inv_temp = 1.0 / float(temperature)
    n_blocks = triton.cdiv(n_cols, block)
    reduce_block = _next_pow2(n_blocks)

    # temp buffers: [M, n_blocks]
    # keep contiguous layout to make stride simple
    pm = torch.empty((M, n_blocks), device=logits.device, dtype=torch.float32)
    m = torch.empty((M,), device=logits.device, dtype=torch.float32)
    ps = torch.empty((M, n_blocks), device=logits.device, dtype=torch.float32)
    psx = torch.empty((M, n_blocks), device=logits.device, dtype=torch.float32)
    out = torch.empty((M,), device=logits.device, dtype=torch.float32)

    grid2d = (M, n_blocks)
    grid1d = (M,)

    _partial_max_kernel[grid2d](
        x2d,
        pm,
        x2d.stride(0),
        x2d.stride(1),
        pm.stride(0),
        pm.stride(1),
        n_cols,
        inv_temp,
        BLOCK=block,
        num_warps=num_warps,
    )

    _reduce_max_kernel[grid1d](
        pm,
        m,
        pm.stride(0),
        pm.stride(1),
        n_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=4 if reduce_block <= 128 else 8,
    )

    _partial_sums_kernel[grid2d](
        x2d,
        m,
        ps,
        psx,
        x2d.stride(0),
        x2d.stride(1),
        ps.stride(0),
        ps.stride(1),
        n_cols,
        inv_temp,
        BLOCK=block,
        num_warps=num_warps,
    )

    _reduce_sums_kernel[grid1d](
        m,
        ps,
        psx,
        out,
        ps.stride(0),
        ps.stride(1),
        n_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=4 if reduce_block <= 128 else 8,
    )

    return out.reshape(logits.shape[:-1])


# ------------------------- PyTorch Baselines -------------------------


def entropy_torch_naive(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    x = logits / temperature
    p = F.softmax(x, dim=-1)
    lp = F.log_softmax(x, dim=-1)
    return -(p * lp).sum(dim=-1)


def entropy_torch_optimized(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # H = logZ - E_p[x]
    x = logits / temperature
    m = x.max(dim=-1, keepdim=True).values
    y = (x - m).exp()
    s = y.sum(dim=-1, keepdim=True)
    logZ = s.log() + m
    ex = (y * x).sum(dim=-1, keepdim=True) / s
    return (logZ - ex).squeeze(-1).to(torch.float32)


# ------------------------- Benchmark Helpers -------------------------


@torch.no_grad()
def bench_fn(fn, x, temperature: float, warmup: int, iters: int) -> float:
    # returns average milliseconds per call
    # warmup
    for _ in range(warmup):
        y = fn(x, temperature)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = fn(x, temperature)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab", type=int, default=151936, help="Qwen3 vocab size (default 151936)"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"]
    )
    parser.add_argument(
        "--B", type=int, nargs="*", default=[1, 2, 4], help="batch sizes to test"
    )
    parser.add_argument(
        "--S",
        type=int,
        nargs="*",
        default=[1, 8, 16, 32],
        help="sequence lengths to test",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--block",
        type=int,
        default=2048,
        choices=[1024, 2048],
        help="Triton tile along vocab",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    device = "cuda"
    V = args.vocab
    T = args.temperature

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Vocab: {V} | temperature: {T} | dtype: {args.dtype}")
    print(f"Triton BLOCK: {args.block}")
    print("-" * 80)

    # compile/warmup once with a representative small shape
    x_warm = torch.randn((max(args.B) * max(args.S), V), device=device, dtype=dtype)
    _ = entropy_triton_qwen3(x_warm, T, block=args.block)
    torch.cuda.synchronize()

    # header
    print(
        f"{'B':>3} {'S':>3} {'M=B*S':>6} | {'torch_naive(ms)':>14} {'torch_opt(ms)':>13} {'triton(ms)':>10} | {'diff(triton vs opt)':>18}"
    )
    print("-" * 80)

    for B in args.B:
        for S in args.S:
            M = B * S
            x = torch.randn((M, V), device=device, dtype=dtype)

            # correctness check vs optimized torch (more stable and closer to our math)
            ref = entropy_torch_optimized(x, T)
            tri = entropy_triton_qwen3(x, T, block=args.block)
            diff = max_abs_diff(tri, ref)

            # benchmark
            ms_naive = bench_fn(entropy_torch_naive, x, T, args.warmup, args.repeats)
            ms_opt = bench_fn(entropy_torch_optimized, x, T, args.warmup, args.repeats)
            ms_tri = bench_fn(
                lambda a, t: entropy_triton_qwen3(a, t, block=args.block),
                x,
                T,
                args.warmup,
                args.repeats,
            )

            print(
                f"{B:>3} {S:>3} {M:>6} | {ms_naive:>14.4f} {ms_opt:>13.4f} {ms_tri:>10.4f} | {diff:>18.3e}"
            )

    print("-" * 80)
    print("Notes:")
    print(
        "1) small-batch sampling often uses M=B*S small; this script benchmarks that directly."
    )
    print(
        "2) triton output is fp32; torch_opt also returns fp32; compare diff accordingly."
    )


if __name__ == "__main__":
    main()
