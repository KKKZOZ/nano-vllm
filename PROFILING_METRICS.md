# Hybrid Generation Profiling Metrics

本文档详细说明 Entropy-based Routing Strategy 生成的统计指标含义。

Example:

```json
Statatics: {
  "total_tokens": 1000,
  "slm_tokens": 793,
  "llm_tokens": 207,
  "decode_steps": 1000,
  "slm_llm_top1_set_equal": 103,
  "slm_llm_top2_set_equal": 122,
  "slm_llm_top3_set_equal": 101,
  "slm_llm_top1_exact": 103,
  "slm_llm_top2_exact": 72,
  "slm_llm_top3_exact": 46,
  "top2_set_equal_avg_cumprob": {
    "slm": 0.9789,
    "llm": 0.9833,
    "min": 0.9695
  },
  "top3_set_equal_avg_cumprob": {
    "slm": 0.9971,
    "llm": 0.9951,
    "min": 0.9928
  },
  "topp_pool_set_equal": 68,
  "topp_pool_set_equal_avg_cumprob": {
    "slm": 0.9916,
    "llm": 0.9913,
    "min": 0.9872
  },
  "topp_pool_sizes": {
    "slm_avg": 2.26,
    "llm_avg": 2.0,
    "when_equal_avg": 2.06
  },
  "elapsed_time": 7.890995740890503,
  "threshold": 0.2,
  "llm_consecutive_tokens": 1
}
```


## 目录

- [基础统计](#基础统计)
- [Top-k 候选集统计](#top-k-候选集统计)
- [累计概率统计](#累计概率统计)
- [Top-p 采样池统计](#top-p-采样池统计)
- [完整示例解读](#完整示例解读)

---

## 基础统计

### `total_tokens`
**定义**: 总共生成的 token 数量（不包括 prompt）

**示例**:
```json
"total_tokens": 1000
```
生成了 1000 个新 token。

---

### `slm_tokens` / `llm_tokens`
**定义**:
- `slm_tokens`: SLM（小模型）生成的 token 数量
- `llm_tokens`: LLM（大模型）生成的 token 数量

**关系**: `slm_tokens + llm_tokens = total_tokens`

**示例**:
```json
"slm_tokens": 793,
"llm_tokens": 207
```
- SLM 生成了 793 个 token（79.3%）
- LLM 生成了 207 个 token（20.7%）

---

### `decode_steps`
**定义**: 总的解码步数（每次生成一个 token 算一步）

**示例**:
```json
"decode_steps": 1000
```
通常等于 `total_tokens`，但在某些策略下可能不同。

---

### `threshold`
**定义**: 熵阈值，当 SLM 的输出熵 ≥ threshold 时切换到 LLM

**示例**:
```json
"threshold": 0.2
```
当 SLM 的熵超过 0.2 时，认为 SLM "不确定"，切换到 LLM 生成。

---

### `elapsed_time`
**定义**: 生成过程的总耗时（秒）

**示例**:
```json
"elapsed_time": 7.89
```
总共耗时 7.89 秒，生成速度 ≈ 1000/7.89 = 126.7 tokens/s

---

## Top-k 候选集统计

当 SLM 因熵高而切换到 LLM 时，我们比较两个模型对**同一位置**的预测。

### `slm_llm_top{k}_set_equal`
**定义**: SLM 和 LLM 的 top-k 候选**集合相同**（不考虑顺序）的次数

**示例场景**:
```
某个位置：
SLM 预测: top-2 = [token_A, token_B]  P(A)=0.6, P(B)=0.4
LLM 预测: top-2 = [token_B, token_A]  P(B)=0.55, P(A)=0.45

集合: {A, B} == {B, A} ✓
→ slm_llm_top2_set_equal +1

但 top-1 不同: A ≠ B ✗
→ slm_llm_top1_set_equal 不计数
```

**解读**:
```json
"slm_llm_top1_set_equal": 103,  // 207 次中有 103 次（49.8%）最优选择相同
"slm_llm_top2_set_equal": 122,  // 207 次中有 122 次（59.0%）前 2 候选集合相同
"slm_llm_top3_set_equal": 101   // 207 次中有 101 次（48.8%）前 3 候选集合相同
```

**注意**: top2_set_equal 可能 > top1_set_equal，因为集合相同不代表排序相同。

---

### `slm_llm_top{k}_exact`
**定义**: SLM 和 LLM 的 top-k 候选**完全相同**（考虑顺序）的次数

**示例场景**:
```
某个位置：
SLM 预测: top-3 = [A, B, C]  P(A)=0.5, P(B)=0.3, P(C)=0.2
LLM 预测: top-3 = [A, B, C]  P(A)=0.48, P(B)=0.32, P(C)=0.18

完全相同: [A,B,C] == [A,B,C] ✓
→ slm_llm_top3_exact +1
→ slm_llm_top2_exact +1 (前 2 个也相同)
→ slm_llm_top1_exact +1 (top-1 也相同)
```

**解读**:
```json
"slm_llm_top1_exact": 103,
"slm_llm_top2_exact": 72,
"slm_llm_top3_exact": 46
```

满足单调性: `top3_exact ≤ top2_exact ≤ top1_exact`

- 46 次（22.2%）前 3 个候选完全相同（含顺序）→ 概率分布非常接近
- 72 次（34.8%）前 2 个候选完全相同
- 103 次（49.8%）最优选择相同

---

## 累计概率统计

### `top{k}_set_equal_avg_cumprob`
**定义**: 在 top-k 候选集合相同的情况下，这 k 个候选的**平均累计概率**

**为什么重要**: 候选集相同不等于会采样到相同结果，还要看这些候选占据多少概率质量。

**示例场景**:
```
某次 top-2 set_equal：
SLM: top-2 = [A, B]
  P(A) = 0.65
  P(B) = 0.32
  累计概率 = 0.97

LLM: top-2 = [B, A]
  P(B) = 0.55
  P(A) = 0.43
  累计概率 = 0.98

记录:
  slm_cumprob = 0.97
  llm_cumprob = 0.98
  min_cumprob = 0.97
```

对所有 122 次 top-2 set_equal 取平均：

**解读**:
```json
"top2_set_equal_avg_cumprob": {
  "slm": 0.9789,  // SLM 在这些位置的平均累计概率
  "llm": 0.9833,  // LLM 在这些位置的平均累计概率
  "min": 0.9695   // 每次取两者中较小值的平均（保守估计）
}
```

**含义**:
- 在前 2 候选集合相同的 122 次中
- 平均有 **96.95%** 的概率会从这 2 个候选中采样
- 只有 **3.05%** 的概率采样到其他 token
- **结论**: 两个模型很可能会采样到相同的 token

---

### `topp_pool_set_equal_avg_cumprob`
**定义**: 在 top-p 采样池相同的情况下，这些候选的平均累计概率

**特殊性**: 由于 top-p=0.95，理论上累计概率应该 ≥ 0.95，但实际会更高

**为什么 > 0.95**:
```
Top-p 算法会包含"跨越阈值"的 token：

Token A: P=0.60  累计=0.60 < 0.95
Token B: P=0.30  累计=0.90 < 0.95
Token C: P=0.08  累计=0.98 > 0.95 ← 跨越了！

候选池 = {A, B, C}
累计概率 = 0.98（而不是 0.95）
```

**解读**:
```json
"topp_pool_set_equal_avg_cumprob": {
  "slm": 0.9916,
  "llm": 0.9913,
  "min": 0.9872
}
```

**含义**:
- 在采样池相同的 68 次中
- 平均有 **98.72%** 的概率从这个池中采样
- 只有 **1.28%** 的概率选择池外的 token
- **结论**: 几乎肯定会采样到相同的 token

---

## Top-p 采样池统计

这些指标反映了**实际采样过程**（应用 temperature 和 top_p 参数后）。

### `topp_pool_set_equal`
**定义**: 应用 top_p=0.95 后，SLM 和 LLM 的采样池**完全相同**的次数

**示例场景**:
```
某个位置（temperature=0.6, top_p=0.95）:

SLM 概率分布（应用 temperature 后）:
  A: 0.55  累计=0.55
  B: 0.30  累计=0.85
  C: 0.10  累计=0.95 ← 刚好达到！
  D: 0.03  累计=0.98
  → 候选池 = {A, B, C}

LLM 概率分布（应用 temperature 后）:
  B: 0.50  累计=0.50
  A: 0.40  累计=0.90
  C: 0.08  累计=0.98 ← 跨越阈值
  → 候选池 = {B, A, C} = {A, B, C}

候选池相同 ✓
→ topp_pool_set_equal +1
```

**解读**:
```json
"llm_tokens": 207,
"topp_pool_set_equal": 68
```
在 207 次 LLM 生成中，有 68 次（32.9%）采样池完全相同。

---

### `topp_pool_sizes`
**定义**: Top-p 采样池的大小统计

**三个指标**:
- `slm_avg`: SLM 采样池的平均大小（所有 207 次）
- `llm_avg`: LLM 采样池的平均大小（所有 207 次）
- `when_equal_avg`: 采样池相同时的平均大小（68 次）

**示例场景**:
```
某次比较：
SLM 采样池 = {A, B, C, D, E}  大小=5
LLM 采样池 = {A, B, C}        大小=3
→ 不相同，但记录 slm=5, llm=3

另一次比较：
SLM 采样池 = {A, B}           大小=2
LLM 采样池 = {A, B}           大小=2
→ 相同！记录 when_equal=2
```

**解读**:
```json
"topp_pool_sizes": {
  "slm_avg": 2.26,
  "llm_avg": 2.0,
  "when_equal_avg": 2.06
}
```

**含义**:
- SLM 平均候选池有 **2.26 个 token**
- LLM 平均候选池有 **2.0 个 token**
- 当候选池相同时，平均有 **2.06 个 token**

**为什么 when_equal_avg 较小**:
- 候选池更可能在**概率分布集中**时相同
- 概率集中 → 少数几个 token 占据大部分概率 → 候选池小
- 概率分散 → 需要更多 token 才能达到 95% → 候选池大 → 不容易完全相同

**采样相同的概率估计**:
```
假设 when_equal_avg = 2.06，候选池相同：

粗略估计（如果概率均匀）:
  P(采样到相同 token) ≈ 1/2.06 ≈ 48.5%

实际情况（概率倾斜，top-1 占主导）:
  P(采样到相同 token) > 50%（可能到 60-70%）
```

---

## 完整示例解读

```json
{
  "total_tokens": 1000,
  "slm_tokens": 793,
  "llm_tokens": 207,
  "decode_steps": 1000,
  "slm_llm_top1_set_equal": 103,
  "slm_llm_top2_set_equal": 122,
  "slm_llm_top3_set_equal": 101,
  "slm_llm_top1_exact": 103,
  "slm_llm_top2_exact": 72,
  "slm_llm_top3_exact": 46,
  "top2_set_equal_avg_cumprob": {
    "slm": 0.9789,
    "llm": 0.9833,
    "min": 0.9695
  },
  "top3_set_equal_avg_cumprob": {
    "slm": 0.9971,
    "llm": 0.9951,
    "min": 0.9928
  },
  "topp_pool_set_equal": 68,
  "topp_pool_set_equal_avg_cumprob": {
    "slm": 0.9916,
    "llm": 0.9913,
    "min": 0.9872
  },
  "topp_pool_sizes": {
    "slm_avg": 2.26,
    "llm_avg": 2.0,
    "when_equal_avg": 2.06
  },
  "elapsed_time": 7.89,
  "threshold": 0.2,
  "llm_consecutive_tokens": 1
}
```

### 关键发现

#### 1. 基础效率
- ✅ **SLM 使用率 79.3%**: 大部分生成由小模型完成
- ✅ **生成速度 126.7 tok/s**: 性能良好
- ✅ **LLM 调用 207 次**: 仅在不确定时使用大模型

#### 2. 候选一致性分析

**Top-1 一致性（49.8%）**:
- 103/207 次最优选择相同
- 说明约一半情况下，SLM 的最佳猜测和 LLM 一致

**Top-2 候选集相同（59.0%）**:
- 122/207 次前 2 候选集合相同
- 累计概率 96.95%
- **结论**: 在这 122 次中，97% 的概率会从相同的 2 个候选中选择

**Top-3 候选集相同（48.8%）**:
- 101/207 次前 3 候选集合相同
- 累计概率 99.28%
- **结论**: 在这 101 次中，几乎肯定会从相同的 3 个候选中选择

#### 3. 实际采样池分析（最重要！）

**候选池相同率（32.9%）**:
- 68/207 次 top-p 采样池完全相同
- 平均候选池大小: **2.06 个 token**
- 累计概率: **98.72%**

**核心结论**:
```
在 32.9% 的 LLM 调用中：
✓ 采样池完全相同（平均 2 个候选）
✓ 这些候选占据 98.72% 的概率
✓ 采样到相同 token 的概率 > 50%
```

#### 4. 整体评估

虽然 SLM 在这些位置"不确定"（熵 ≥ 0.2），但：

- ✅ **49.8%** 的情况最优选择和 LLM 一致
- ✅ **59.0%** 的情况前 2 候选和 LLM 一致（97% 概率从中选）
- ✅ **32.9%** 的情况采样池和 LLM 完全相同

**潜在优化方向**:
1. 对于 top-1 相同的情况，可以考虑直接用 SLM 的预测
2. 对于候选池相同的情况，可以考虑更激进的路由策略
3. 调整熵阈值以平衡质量和效率

---

## 统计指标关系图

```
LLM 调用 207 次
    │
    ├─ Top-1 相同: 103 次 (49.8%)
    │   └─ 最优选择一致
    │
    ├─ Top-2 集合相同: 122 次 (59.0%)
    │   ├─ 其中完全相同（含顺序）: 72 次 (34.8%)
    │   └─ 累计概率: 96.95%
    │
    ├─ Top-3 集合相同: 101 次 (48.8%)
    │   ├─ 其中完全相同（含顺序）: 46 次 (22.2%)
    │   └─ 累计概率: 99.28%
    │
    └─ Top-p 采样池相同: 68 次 (32.9%)
        ├─ 平均候选数: 2.06 个
        ├─ 累计概率: 98.72%
        └─ 采样相同概率: > 50%
```

---

## 使用建议

### 1. 评估路由质量
- 查看 `top1_set_equal` 占比：高 → SLM 判断准确
- 查看累计概率：高 → 候选范围一致

### 2. 优化阈值
- 如果 `topp_pool_set_equal` 占比高 → 可以提高阈值（减少 LLM 调用）
- 如果 `top1_set_equal` 占比低 → 阈值设置合理

### 3. 分析采样行为
- `topp_pool_sizes` 小 → 概率分布集中 → 更容易生成相同结果
- `topp_pool_sizes` 大 → 概率分布分散 → 可能需要调整 temperature

---

## 常见问题

### Q1: 为什么 top2_set_equal > top1_set_equal？
A: 因为 set_equal 不考虑顺序。集合 {A,B} = {B,A}，但 top-1 不同（A ≠ B）。

### Q2: 为什么累计概率 > 0.95（设置的 top_p）？
A: Top-p 算法会包含"跨越阈值"的 token，所以实际累计概率通常略高于设定值。

### Q3: 如何判断两个模型是否会生成相同结果？
A: 主要看 `topp_pool_set_equal` 和 `topp_pool_sizes.when_equal_avg`：
- 候选池相同 + 候选数少 + 累计概率高 → 很可能生成相同

### Q4: 为什么 when_equal_avg < slm_avg？
A: 候选池更容易在概率集中时相同，而概率集中意味着候选数少。

---

## 参考

- 代码实现: `hybrid_generator/strategies/route.py`
- 生成策略: Entropy-based Routing
- 采样参数: temperature=0.6, top_k=20, top_p=0.95
