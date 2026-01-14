import time

import numpy as np
import torch
import torch.nn.functional as F


# === 你的目标函数 ===
def calculate_token_entropy(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Calculate token entropy from logits.
    """
    # Handle temperature = 0 case
    if temperature == 0.0:
        if logits.dim() == 1:
            return torch.tensor(0.0, device=logits.device)
        else:
            return torch.zeros(logits.shape[0], device=logits.device)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Calculate probability distribution (Softmax)
    probs = F.softmax(scaled_logits, dim=-1)

    # Calculate entropy H = -sum(p * log(p))
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


# === 性能计时器辅助类 ===
class CUDATimer:
    def __init__(self, name):
        self.name = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.cpu_start = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self.cpu_start = time.time()
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_cuda = self.start_event.elapsed_time(self.end_event)  # ms
        elapsed_cpu = (time.time() - self.cpu_start) * 1000  # ms
        # 这里的返回不直接打印，由外部获取数据


def benchmark_function(func, inputs, input_args, iterations=100, label="Method"):
    # 1. Warmup / Compilation (First Run)
    print(f"--- 测试: {label} ---")

    # 强制同步清理
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 捕捉第一次运行时间 (对于 compile 来说就是编译耗时)
    with CUDATimer("First Run") as t:
        _ = func(inputs, *input_args)

    first_run_time = t.start_event.elapsed_time(t.end_event)
    print(f"[{label}] 首次运行 (含编译/Warmup): {first_run_time:.2f} ms")

    # 2. 稳定性测试 (Average Latency)
    # 预跑几次稳定一下 GPU 频率
    for _ in range(5):
        _ = func(inputs, *input_args)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = func(inputs, *input_args)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    avg_time = np.mean(times)
    p99_time = np.percentile(times, 99)
    print(f"[{label}] 平均推理耗时 (Avg): {avg_time:.4f} ms")
    print(f"[{label}] P99 推理耗时: {p99_time:.4f} ms")
    print("-" * 30)
    return first_run_time, avg_time


# === 主程序 ===
if __name__ == "__main__":
    # 配置参数
    BATCH_SIZE = 1
    VOCAB_SIZE = 151936  # 类似 Llama 2 的词表大小
    TEMP = 0.7
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"运行环境: {DEVICE.upper()}")
    print(f"输入形状: ({BATCH_SIZE}, {VOCAB_SIZE}) | float16")

    # 准备数据 (模拟 LLM Logits)
    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, device=DEVICE, dtype=torch.float16)

    # 1. 测试 Eager Mode (原生)
    eager_first, eager_avg = benchmark_function(
        calculate_token_entropy, logits, (TEMP,), label="Eager Mode"
    )

    # 2. 测试 torch.compile (Default Mode)
    # 重置编译缓存
    torch.compiler.reset()
    compiled_model_default = torch.compile(calculate_token_entropy)

    compile_default_first, compile_default_avg = benchmark_function(
        compiled_model_default, logits, (TEMP,), label="Compile (Default)"
    )

    # 3. 测试 torch.compile (Reduce Overhead - 适合小 Batch 推理)
    # 注意：reduce-overhead 使用 CUDA Graphs，对形状非常敏感
    torch.compiler.reset()
    try:
        compiled_model_ro = torch.compile(
            calculate_token_entropy, mode="reduce-overhead"
        )
        compile_ro_first, compile_ro_avg = benchmark_function(
            compiled_model_ro, logits, (TEMP,), label="Compile (Reduce-Overhead)"
        )
    except Exception as e:
        print(f"Reduce-Overhead 模式运行失败 (可能显存不足或不支持): {e}")
        compile_ro_avg = eager_avg  # Fallback

    # === 总结 ===
    print("\n=== 结果汇总 ===")
    print(f"Eager 平均耗时:   {eager_avg:.4f} ms")
    print(
        f"Default 编译开销: {compile_default_first:.2f} ms (比 Eager 慢 {compile_default_first / eager_first:.1f}x)"
    )
    print(f"Default 加速比:   {eager_avg / compile_default_avg:.2f}x")
    if "compile_ro_avg" in locals():
        print(f"R-Overhead 加速比:{eager_avg / compile_ro_avg:.2f}x")
