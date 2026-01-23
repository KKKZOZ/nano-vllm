# Async Entropy-Based Routing Strategy

## 概述

`AsyncEntropyStrategy` 实现了一个支持并发执行的熵值路由策略。当 SLM 检测到高熵（不确定性）时，它会异步触发 LLM 生成，同时继续推测性地生成 tokens。

## 核心特性

### 1. 并发执行
- SLM 和 LLM 可以在不同 GPU 上并发运行
- 当 SLM 遇到高熵时，触发 LLM 异步任务
- SLM 继续推测性生成，无需等待 LLM

### 2. 智能停止条件

SLM 在以下两种情况下停止推测：

- **条件 1**: SLM 再次遇到高熵 token（需要与 LLM 同步）
- **条件 2**: LLM 完成生成（需要验证）

### 3. 验证机制

在分歧点比较 SLM 和 LLM 的 logits：

- 检查 top-3 token IDs 是否匹配（作为集合）
- 检查累积概率是否 >= 0.8
- **一致**: 接受 SLM 推测的所有 tokens
- **不一致**: 回滚 SLM，使用 LLM 的 token

### 4. 后端兼容性

- **同步后端**: 使用 `ThreadPoolExecutor` 包装（如 HFBackend）
- **异步后端**: 直接支持（未来的 HTTP API 后端）

## 架构设计

### 核心组件

```
AsyncBackendWrapper (async_wrapper.py)
├─ 包装同步后端提供异步接口
├─ 使用 ThreadPoolExecutor 在后台线程运行
└─ 线程安全的 CUDA 上下文管理

AsyncHybridBackend (hybrid.py)
├─ 扩展 HybridBackend 支持异步操作
├─ 包装 SLM 和 LLM 后端
└─ 提供 forward_async() 方法

AsyncEntropyStrategy (async_route.py)
├─ 主要策略实现
├─ SpeculativeBuffer: 追踪推测性 tokens
├─ LLMResult: LLM 异步任务结果
├─ 验证逻辑
└─ 回滚处理
```

### 工作流程

```
1. Prefill 阶段（同步）
   ├─ 用 prompt tokens 预填充 SLM 和 LLM
   └─ 获取初始 SLM logits

2. 主生成循环
   │
   ├─ Case 1: 低熵 & 无 LLM 任务
   │   ├─ 从 SLM 采样
   │   ├─ 在 SLM 中前向传播 1 个 token
   │   └─ 更新 slm_synced_len
   │
   ├─ Case 2: 高熵检测 & 无 LLM 任务
   │   ├─ 存储 divergence_pos 和 divergence_logits
   │   ├─ 创建异步任务: _llm_generate_from_divergence()
   │   └─ 继续下一次迭代（SLM 开始推测）
   │
   └─ Case 3: LLM 任务运行中（SLM 推测）
       │
       ├─ 如果 LLM 完成:
       │   ├─ 等待 LLM 结果
       │   ├─ 在分歧点验证 logits
       │   ├─ 如果验证通过: 接受推测 tokens
       │   └─ 如果失败: 回滚 SLM，使用 LLM token
       │
       └─ 如果 LLM 未完成:
           ├─ 检查熵
           ├─ 如果 entropy >= threshold: 强制等待 LLM（条件 1）
           └─ 否则: 生成推测性 token，存入 buffer

3. 验证阶段
   ├─ 应用温度缩放
   ├─ 获取 top-3 tokens
   ├─ 检查 top-3 集合匹配
   ├─ 检查累积概率 >= 0.8
   └─ 返回 accept/reject + 统计信息

4. 回滚处理（验证失败时）
   ├─ 回滚 SLM 到 divergence_pos
   ├─ 将 LLM 的 token 喂给 SLM
   └─ 更新 slm_synced_len
```

