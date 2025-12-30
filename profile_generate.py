"""
Profile the generate_with_profile function using PyTorch Profiler.

Focus areas:
1. SLM inference (self.slm.forward)
2. Entropy calculation (calculate_token_entropy, compute_logu)
3. Sampling (sample_token)
4. Other operations (topk, add_token, tokenizer decode, etc.)
"""

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
)
from transformers import AutoTokenizer

from hybrid_generator import HybridGenerator


def run_profiling():
    # Configuration
    draft_model = "/root/huggingface/Qwen3-1.7B"
    model = "/root/huggingface/Qwen3-8B"

    input_text = "Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text},
    ]

    tokenizer = AutoTokenizer.from_pretrained(draft_model)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Initialize generator
    print("Initializing generator...")
    generator = HybridGenerator(
        slm_model_id=draft_model,
        llm_model_id=None,  # Only SLM for profiling
        slm_memory_usage=0.4,
        device="cuda",
        dtype=torch.float16,
        verbose=False,
        enable_stats_sync=True,  # Important for accurate timing
    )

    # Warmup run (important for CUDA graphs and JIT)
    print("Warmup run...")
    _ = generator.generate_with_profile(prompt=prompt, max_new_tokens=10)
    torch.cuda.synchronize()

    # Profile run
    print("\nStarting profiling...")
    max_tokens = 100  # Adjust based on your needs

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # Enable call stack for better tracing
        with_flops=True,  # Estimate FLOPs
    ) as prof:
        result = generator.generate_with_profile(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        )

    torch.cuda.synchronize()

    # Print results sorted by CUDA time
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - Sorted by CUDA Time (Total)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=50,
    ))

    # Print results sorted by CPU time
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - Sorted by CPU Time (Total)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=30,
    ))

    # Print results grouped by input shapes
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - Grouped by Input Shapes")
    print("=" * 80)
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=30,
    ))

    # Export Chrome trace for visualization
    trace_path = "profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("Open chrome://tracing in Chrome browser to visualize")

    # Custom analysis: Group operations by category
    print("\n" + "=" * 80)
    print("CUSTOM ANALYSIS - Operations by Category")
    print("=" * 80)

    # More specific categorization based on actual kernel names from profiling output
    categories = {
        "SLM Inference (GEMM/GEMV)": [
            "gemm", "gemv", "matmul", "mm", "linear", "addmm", "bmm",
        ],
        "SLM Inference (Attention)": [
            "flash", "attention", "splitkv", "kvcache",
        ],
        "SLM Inference (Norm/Activ)": [
            "rmsnorm", "layernorm", "silu", "rsqrt", "pow", "mean",
        ],
        "Entropy/Softmax": [
            "softmax", "_softmax", "cunn_softmax",
        ],
        "Sampling (TopK/Sort)": [
            "topk", "sort", "radix", "mbtopk",
        ],
        "Sampling (Multinomial)": [
            "multinomial", "cumsum", "searchsorted",
        ],
        "Other Compute": [
            "sum", "log", "exp", "mul", "add", "div",
        ],
        "Memory/Copy": [
            "copy", "memcpy", "to", "clone", "contiguous",
        ],
        "Misc": [],
    }

    category_times = {cat: {"cpu": 0, "cuda": 0, "count": 0} for cat in categories}

    for event in prof.key_averages():
        name_lower = event.key.lower()
        matched = False

        # Get CUDA time - try different attribute names
        cuda_time = 0
        for attr in ['self_cuda_time_total', 'cuda_time_total', 'device_time_total', 'self_device_time_total']:
            if hasattr(event, attr):
                cuda_time = getattr(event, attr, 0)
                if cuda_time > 0:
                    break

        cpu_time = event.cpu_time_total
        count = event.count

        for cat, keywords in categories.items():
            if cat == "Misc":
                continue
            for keyword in keywords:
                if keyword.lower() in name_lower:
                    category_times[cat]["cpu"] += cpu_time
                    category_times[cat]["cuda"] += cuda_time
                    category_times[cat]["count"] += count
                    matched = True
                    break
            if matched:
                break

        if not matched:
            category_times["Misc"]["cpu"] += cpu_time
            category_times["Misc"]["cuda"] += cuda_time
            category_times["Misc"]["count"] += count

    # Calculate totals
    total_cpu = sum(c["cpu"] for c in category_times.values())
    total_cuda = sum(c["cuda"] for c in category_times.values())

    print(f"\n{'Category':<30} {'CPU Time':>12} {'CPU %':>8} {'CUDA Time':>12} {'CUDA %':>8} {'Calls':>8}")
    print("-" * 80)

    for cat, times in sorted(category_times.items(), key=lambda x: x[1]["cuda"], reverse=True):
        cpu_pct = times["cpu"] / total_cpu * 100 if total_cpu > 0 else 0
        cuda_pct = times["cuda"] / total_cuda * 100 if total_cuda > 0 else 0
        print(f"{cat:<30} {times['cpu']/1000:>10.2f}ms {cpu_pct:>7.1f}% {times['cuda']/1000:>10.2f}ms {cuda_pct:>7.1f}% {times['count']:>8}")

    print("-" * 80)
    print(f"{'TOTAL':<30} {total_cpu/1000:>10.2f}ms {100:>7.1f}% {total_cuda/1000:>10.2f}ms {100:>7.1f}%")

    # Print summary for user's 4 categories
    print("\n" + "=" * 80)
    print("SUMMARY - Your 4 Categories (CUDA Time)")
    print("=" * 80)

    slm_inference = (
        category_times["SLM Inference (GEMM/GEMV)"]["cuda"] +
        category_times["SLM Inference (Attention)"]["cuda"] +
        category_times["SLM Inference (Norm/Activ)"]["cuda"]
    )
    entropy_calc = category_times["Entropy/Softmax"]["cuda"]
    sampling = (
        category_times["Sampling (TopK/Sort)"]["cuda"] +
        category_times["Sampling (Multinomial)"]["cuda"]
    )
    other = (
        category_times["Other Compute"]["cuda"] +
        category_times["Memory/Copy"]["cuda"] +
        category_times["Misc"]["cuda"]
    )

    user_total = slm_inference + entropy_calc + sampling + other

    print(f"\n{'Category':<25} {'CUDA Time':>15} {'Percentage':>12}")
    print("-" * 55)
    print(f"{'1. SLM Inference':<25} {slm_inference/1000:>12.2f}ms {slm_inference/user_total*100 if user_total > 0 else 0:>11.1f}%")
    print(f"{'2. Entropy Calculate':<25} {entropy_calc/1000:>12.2f}ms {entropy_calc/user_total*100 if user_total > 0 else 0:>11.1f}%")
    print(f"{'3. Sampling':<25} {sampling/1000:>12.2f}ms {sampling/user_total*100 if user_total > 0 else 0:>11.1f}%")
    print(f"{'4. Other Operations':<25} {other/1000:>12.2f}ms {other/user_total*100 if user_total > 0 else 0:>11.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<25} {user_total/1000:>12.2f}ms {100:>11.1f}%")

    return prof


if __name__ == "__main__":
    run_profiling()
