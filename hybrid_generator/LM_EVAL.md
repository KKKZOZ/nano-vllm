# HybridLM - LM Evaluation Harness Integration

这个文档说明如何使用 `HybridLM` 通过 lm-evaluation-harness 对混合模型进行标准化评测。

## 安装依赖

```bash
pip install lm-eval>=0.4.0
```

## 快速开始

### 方法 1: 命令行方式 (推荐)

```bash
# 基本用法
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty,threshold=0.5" \
        --tasks hellaswag,arc_easy \
        --device cuda \
        --batch_size 1

# 使用不同策略
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=entropy,threshold=2.0" \
        --tasks winogrande,piqa \
        --device cuda

# Speculative decoding
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=speculative,num_drafts=4" \
        --tasks mmlu \
        --device cuda
```

### 方法 2: Python 代码方式

```python
from lm_eval import evaluator
from hybrid_generator import HybridLM

# 创建模型
model = HybridLM(
    slm_model_id="Qwen/Qwen3-1.7B",
    llm_model_id="Qwen/Qwen3-8B",
    strategy="uncertainty",
    threshold=0.5,
    device="cuda",
)

# 运行评测
results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "arc_easy"],
    batch_size=1,
    num_fewshot=0,  # 0-shot evaluation
)

# 查看结果
for task, metrics in results["results"].items():
    print(f"\n{task}:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
```

## 支持的参数

### Model Args (命令行)

通过 `--model_args` 传递参数,使用逗号分隔:

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `slm` | 小模型 ID (必需) | - | `slm=Qwen/Qwen3-1.7B` |
| `llm` | 大模型 ID (必需) | - | `llm=Qwen/Qwen3-8B` |
| `strategy` | 策略名称 | `uncertainty` | `strategy=entropy` |
| `threshold` | 路由阈值 | `0.5` | `threshold=0.7` |
| `num_drafts` | Draft 数量 (speculative) | `4` | `num_drafts=6` |
| `device` | 设备 | `cuda` | `device=cpu` |
| `dtype` | 数据类型 | `float16` | `dtype=bfloat16` |
| `batch_size` | 批次大小 | `1` | `batch_size=1` |

### Python 初始化参数

```python
HybridLM(
    slm_model_id: str,           # 小模型 ID
    llm_model_id: str,           # 大模型 ID
    strategy: str = "uncertainty",   # 策略: speculative/uncertainty/entropy
    threshold: float = 0.5,      # 路由阈值
    num_drafts: int = 4,         # Speculative 策略的 draft 数
    device: str = "cuda",        # 设备
    dtype: str = "float16",      # 数据类型
    batch_size: int = 1,         # 批次大小
)
```

## 常用评测任务

### 轻量级任务 (快速测试)

```bash
# HellaSwag - 常识推理 (~10k examples)
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks hellaswag \
        --device cuda

# ARC Easy - 小学科学题 (~2.4k examples)
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks arc_easy \
        --device cuda

# WinoGrande - 代词消歧 (~1.3k examples)
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks winogrande \
        --device cuda
```

### 综合基准测试

```bash
# MMLU (Massive Multitask Language Understanding)
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks mmlu \
        --num_fewshot 5 \
        --device cuda

# GSM8K - 数学推理
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks gsm8k \
        --device cuda

# HumanEval - 代码生成
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks humaneval \
        --device cuda
```

## 策略比较

### 示例: 对比不同策略

```python
from lm_eval import evaluator
from hybrid_generator import HybridLM

strategies = [
    ("uncertainty", {"threshold": 0.5}),
    ("entropy", {"threshold": 2.0}),
    ("speculative", {"num_drafts": 4}),
]

tasks = ["hellaswag", "arc_easy"]

for strategy_name, kwargs in strategies:
    print(f"\n{'='*60}")
    print(f"Evaluating: {strategy_name}")
    print(f"{'='*60}")

    model = HybridLM(
        slm_model_id="Qwen/Qwen3-1.7B",
        llm_model_id="Qwen/Qwen3-8B",
        strategy=strategy_name,
        **kwargs
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=1,
    )

    # 打印结果
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
```

### 示例: 阈值敏感度分析

```python
thresholds = [0.3, 0.5, 0.7]
results_dict = {}

for thresh in thresholds:
    model = HybridLM(
        slm_model_id="Qwen/Qwen3-1.7B",
        llm_model_id="Qwen/Qwen3-8B",
        strategy="uncertainty",
        threshold=thresh,
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=["hellaswag"],
        batch_size=1,
    )

    results_dict[thresh] = results

# 对比不同阈值的表现
print("\nThreshold Comparison:")
print(f"{'Threshold':<12} {'Accuracy':<12}")
print("-" * 30)
for thresh, results in results_dict.items():
    acc = results["results"]["hellaswag"]["acc"]
    print(f"{thresh:<12.2f} {acc:<12.4f}")
```

## HybridLM 实现细节

### 核心方法

`HybridLM` 实现了 lm-eval 要求的三个核心方法:

#### 1. `loglikelihood()`
计算给定上下文和续写的对数似然。用于大多数多选题任务。

```python
def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
    """
    Args:
        requests: [(context, continuation), ...]

    Returns:
        [(log_likelihood, is_greedy), ...]
    """
    # 对每个 (context, continuation) 对:
    # 1. 拼接为完整序列
    # 2. 通过 LLM 获取 logits
    # 3. 计算 continuation 部分的 log probability
    # 4. 检查是否与 greedy decoding 一致
```

#### 2. `generate_until()`
生成文本直到满足停止条件。用于生成类任务。

```python
def generate_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
    """
    Args:
        requests: [(context, gen_kwargs), ...]
        gen_kwargs: {
            "until": [停止字符串列表],
            "max_gen_toks": 最大生成 token 数,
            "temperature": 温度,
            "do_sample": 是否采样
        }

    Returns:
        [生成的文本, ...]
    """
    # 对每个请求:
    # 1. 从 context 开始生成
    # 2. 检查停止条件
    # 3. 返回生成的部分(不包含 context)
```

#### 3. `loglikelihood_rolling()`
计算序列的滚动对数似然。用于困惑度评测。

```python
def loglikelihood_rolling(self, requests: List[Tuple[str,]]) -> List[Tuple[float,]]:
    """
    Args:
        requests: [(text,), ...]

    Returns:
        [(total_log_likelihood,), ...]
    """
    # 对每个文本:
    # 1. Tokenize
    # 2. 对每个位置计算 log P(token_i | tokens_0..i-1)
    # 3. 求和
```

### 注意事项

1. **批次大小**: 当前实现仅支持 `batch_size=1`,因为混合生成的动态路由难以批处理

2. **评测模型选择**: `loglikelihood` 方法使用 LLM 进行评分,以确保评测准确性

3. **生成方法**: `generate_until` 目前使用简单的 LLM greedy/sampling 生成,未使用混合策略(可根据需要修改)

4. **性能权衡**:
   - Uncertainty/Entropy 策略: 可能加速,但评测时主要用 LLM
   - Speculative 策略: 理论上可加速,但当前实现未优化批处理

## 完整示例

参考 `example_lm_eval.py` 获取完整示例代码,包括:

- 单策略评测
- 多策略对比
- 阈值敏感度分析
- 结果可视化

运行示例:

```bash
python example_lm_eval.py
```

## 常见问题

### Q: 为什么评测很慢?

A: 当前实现 `batch_size=1`,且每个样本都需要完整的模型前向传播。这是混合策略的固有限制。

### Q: 如何提高评测速度?

A:
1. 使用较小的测试集子集进行快速测试
2. 选择轻量级任务(如 arc_easy, hellaswag)
3. 使用更小的模型

### Q: 评测结果与单独使用 LLM 有差异吗?

A: 在 `loglikelihood` 任务中,我们使用 LLM 计算概率,所以结果应该接近。在生成任务中可能有差异,取决于实现。

### Q: 可以评测中文任务吗?

A: 可以,只要使用的模型支持中文,lm-eval 也支持中文任务。

## 高级用法

### 自定义评测任务

```python
from lm_eval.tasks import TaskManager

# 加载自定义任务
task_manager = TaskManager()

# 使用自定义任务
results = evaluator.simple_evaluate(
    model=model,
    tasks=["your_custom_task"],
    task_manager=task_manager,
)
```

### 保存详细日志

```bash
lm_eval --model hybrid \
        --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \
        --tasks hellaswag \
        --output_path ./results/ \
        --log_samples
```

## 参考资源

- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- 支持的任务列表: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
- 文档: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs
