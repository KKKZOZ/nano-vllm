# Hybrid Generator

一个统一的接口,用于组合小型语言模型(SLM)和大型语言模型(LLM)的多种生成策略。

## 模块结构

```
hybrid_generator/
├── __init__.py         # 模块导出接口
├── generator.py        # HybridGenerator 主类
├── strategies.py       # 三种策略的实现
├── utils.py           # 辅助函数(采样、不确定性、熵计算)
└── README.md          # 本文档
```

## 核心组件

### 1. HybridGenerator (generator.py)
主类,提供统一的生成接口。

**职责**:
- 加载和管理 SLM 和 LLM 模型
- 提供统一的 `generate()` 方法
- 协调不同策略的执行
- 打印统计信息

### 2. Strategies (strategies.py)
三种生成策略的实现:

#### SpeculativeStrategy
- **原理**: SLM 生成多个候选 token,LLM 并行验证
- **优势**: 最大化并行计算,加速解码
- **参数**: `num_drafts` - 每轮生成的候选数量

#### UncertaintyStrategy
- **原理**: 基于 aleatoric uncertainty 路由
- **优势**: 在简单部分使用 SLM,复杂部分使用 LLM
- **参数**: `threshold` - uncertainty 阈值(越低越倾向 LLM)

#### EntropyStrategy
- **原理**: 基于输出熵路由
- **优势**: 根据预测的确定性动态选择模型
- **参数**: `threshold` - 熵阈值(越低越倾向 LLM)

### 3. Utilities (utils.py)
辅助函数集合:

- **`sample_token()`**: 支持 temperature, top-k, top-p, min-p 的采样
- **`compute_logu()`**: 计算 aleatoric 和 epistemic uncertainty
- **`calculate_token_entropy()`**: 计算 token 的熵

## 使用示例

### 基本用法

```python
from hybrid_generator import HybridGenerator

# 初始化生成器
generator = HybridGenerator(
    slm_model_id="Qwen/Qwen3-1.7B",
    llm_model_id="Qwen/Qwen3-8B",
    device="cuda",
    dtype=torch.float16
)

# 使用 speculative decoding
result = generator.generate(
    prompt="Write a story about AI",
    strategy="speculative",
    max_new_tokens=200,
    num_drafts=4,
    temperature=0.6,
    top_k=20,
    top_p=0.95
)
```

### 使用不同策略

```python
# Uncertainty-based routing
result = generator.generate(
    prompt="Explain quantum computing",
    strategy="uncertainty",
    threshold=0.5,  # 低阈值 = 更多使用 LLM
    max_new_tokens=200
)

# Entropy-based routing
result = generator.generate(
    prompt="Write code to sort a list",
    strategy="entropy",
    threshold=2.0,  # 低阈值 = 更多使用 LLM
    max_new_tokens=200
)
```

### 高级用法

```python
# 导入策略类进行自定义
from hybrid_generator import (
    HybridGenerator,
    SpeculativeStrategy,
    UncertaintyStrategy
)

generator = HybridGenerator(
    slm_model_id="Qwen/Qwen3-1.7B",
    llm_model_id="Qwen/Qwen3-8B"
)

# 使用不同的采样参数
result = generator.generate(
    prompt="Creative writing task",
    strategy="speculative",
    temperature=0.8,    # 更高的随机性
    top_k=40,           # 更大的候选池
    top_p=0.9,          # 更宽松的 nucleus sampling
    min_p=0.05          # 过滤低概率 token
)
```

## 参数调优指南

### Speculative Strategy
- **`num_drafts`**:
  - 小 (2-3): 保守,接受率高,加速较少
  - 中 (4-5): 平衡性能和接受率
  - 大 (6-8): 激进,可能加速更多但接受率降低

### Uncertainty Strategy
- **`threshold`**:
  - 低 (0.1-0.3): 更多使用 LLM,质量高,速度慢
  - 中 (0.4-0.6): 平衡质量和速度
  - 高 (0.7-1.0): 更多使用 SLM,速度快,质量可能下降

### Entropy Strategy
- **`threshold`**:
  - 低 (1.0-2.0): 更多使用 LLM
  - 中 (2.0-3.0): 平衡
  - 高 (3.0-5.0): 更多使用 SLM

### 采样参数
- **`temperature`**: 0.6-0.8 适合大多数任务
- **`top_k`**: 20-40 是常见选择
- **`top_p`**: 0.9-0.95 平衡多样性和质量
- **`min_p`**: 通常设为 0.0,需要时可设 0.05-0.1

## 输出统计

每次生成后会打印详细统计:

```
============================================================
Strategy: speculative
Time taken: 15.32s
Total tokens generated: 200
Speed: 13.05 tok/s
Decode steps: 42
Draft tokens: 168, Accepted: 128, Acceptance rate: 76.19%
SLM tokens: 128, LLM tokens: 72
============================================================
```

## 设计原则

1. **模块化**: 每个组件职责单一,易于测试和维护
2. **可扩展**: 通过继承 `GenerationStrategy` 轻松添加新策略
3. **统一接口**: 所有策略通过相同的 API 使用
4. **清晰分离**: 工具函数、策略实现、主类分离

## 扩展新策略

如需添加新策略,继承 `GenerationStrategy` 并实现 `generate()` 方法:

```python
from hybrid_generator.strategies import GenerationStrategy

class MyCustomStrategy(GenerationStrategy):
    def generate(self, slm, llm, tokenizer, prompt,
                 max_new_tokens, temperature, top_k,
                 top_p, min_p, device, **kwargs):
        # 实现你的策略
        # 返回 (生成文本, 统计字典)
        pass

# 注册到 HybridGenerator
generator.strategies['custom'] = MyCustomStrategy()
```

## 依赖项

- PyTorch
- Transformers (HuggingFace)
- Python 3.10+

## 性能考虑

1. **内存**: 同时加载两个模型,确保足够 GPU 内存
2. **Cache 管理**: 所有策略都使用 KV cache 优化
3. **同步开销**: Uncertainty 和 Entropy 策略需要保持两个模型 cache 同步

## Profiling 工具

### generate_with_profile()

新增的 `generate_with_profile()` 方法提供详细的生成过程分析:

```python
profile = generator.generate_with_profile(
    prompt="Explain quantum computing",
    max_new_tokens=100,
    threshold=0.5,
    routing_metric="uncertainty"  # 或 "entropy"
)

# 打印详细摘要
profile.print_summary()

# 分析分布
analysis = profile.analyze_distribution("uncertainty", percentiles=[10, 20, 30])
print(f"Top 10% 的 token 不确定性 >= {analysis['percentiles']['top_10%']:.4f}")

# 获取最不确定的 tokens
high_uncertain = profile.get_high_uncertainty_tokens(10)
for token in high_uncertain:
    print(f"位置 {token.position}: '{token.token_text}' "
          f"(不确定性: {token.aleatoric_uncertainty:.4f})")

# 可视化分布 (需要 matplotlib)
profile.plot_distributions("profile.png")

# 导出数据
profile.save_to_file("profile.json")
```

### ProfileResult 对象

`generate_with_profile()` 返回一个 `ProfileResult` 对象,包含:

**数据**:
- `tokens`: 每个 token 的详细信息列表
- `generated_text`: 生成的文本
- `stats`: 聚合统计信息

**每个 Token 的信息**:
- `token_id`: Token ID
- `token_text`: Token 文本
- `position`: 在序列中的位置
- `aleatoric_uncertainty`: Aleatoric 不确定性
- `epistemic_uncertainty`: Epistemic 不确定性
- `entropy`: 熵值
- `model_used`: 使用的模型 ("slm" 或 "llm")
- `top_k_probs`: Top-K 候选概率

**分析方法**:
- `analyze_distribution(metric)`: 分析不确定性/熵的分布
  - 返回: mean, std, min, max, median, percentiles
- `get_high_uncertainty_tokens(top_n)`: 获取最不确定的 tokens
- `print_summary()`: 打印完整摘要
- `plot_distributions()`: 可视化分布
- `save_to_file()`: 导出到 JSON

### 分位数分析示例

```python
# 获取不确定性分布的分位数
analysis = profile.analyze_distribution("uncertainty")

print(f"平均不确定性: {analysis['mean']:.4f}")
print(f"标准差: {analysis['std']:.4f}")
print(f"范围: [{analysis['min']:.4f}, {analysis['max']:.4f}]")

# 从高到低的分位数
print("\n分位数 (从最高值开始):")
print(f"  Top 10%: {analysis['percentiles']['top_10%']:.4f}")
print(f"  Top 20%: {analysis['percentiles']['top_20%']:.4f}")
print(f"  Top 30%: {analysis['percentiles']['top_30%']:.4f}")
print(f"  中位数: {analysis['median']:.4f}")

# Top 10% 的统计
print(f"\nTop 10% tokens:")
print(f"  平均值: {analysis['top_10_percent']['mean']:.4f}")
print(f"  范围: [{analysis['top_10_percent']['min']:.4f}, "
      f"{analysis['top_10_percent']['max']:.4f}]")
```

### 输出示例

```
======================================================================
Profile Summary - Strategy: profiled_uncertainty
======================================================================

Generated 95 tokens in 8.32s
Speed: 11.42 tok/s

Model Usage:
  SLM: 68 tokens (71.6%)
  LLM: 27 tokens (28.4%)

----------------------------------------------------------------------
Uncertainty Distribution:
----------------------------------------------------------------------
  Count: 95
  Mean: 0.3245 ± 0.1823 (std)
  Range: [0.0512, 0.8934]
  Median: 0.2876

  Top 10% tokens:
    Mean: 0.7234, Range: [0.6123, 0.8934]

  Percentiles (from highest):
    top_10%: 0.6123
    top_20%: 0.5012
    top_30%: 0.4234
    ...

----------------------------------------------------------------------
Top 10 Most Uncertain Tokens:
----------------------------------------------------------------------
 1. Pos  23 | "quantum"            | Uncertainty: 0.8934 | Model: LLM
 2. Pos  45 | "entanglement"       | Uncertainty: 0.7821 | Model: LLM
 3. Pos  67 | "superposition"      | Uncertainty: 0.7456 | Model: LLM
 ...
```

### 使用场景

1. **阈值优化**: 测试不同阈值,找到最佳平衡点
2. **模式分析**: 识别 SLM 在哪些类型的内容上不确定
3. **质量评估**: 检查高不确定性的 tokens 是否合理
4. **策略比较**: 对比 uncertainty 和 entropy 路由的效果

完整示例请参考 `example_profiling.py`。

## 许可证

[根据项目需要添加]
