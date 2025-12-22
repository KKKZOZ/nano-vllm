# HybridLM Evaluation - Quick Start Guide

快速上手指南,帮助你立即开始使用 HybridLM 评估工具。

## 安装依赖

```bash
# 安装 lm-evaluation-harness
pip install lm-eval

# 确认已安装必要依赖
pip install torch transformers
```

## 最简单的用法

### 1. 快速测试 (推荐先运行这个)

```bash
cd hybrid_generator
bash examples/eval_quick_test.sh
```

这会在少量样本上测试 HybridLM,确保一切正常工作。

### 2. 基础评估

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --strategy uncertainty \
    --threshold 0.5
```

### 3. 基线对比 (SLM only)

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --use_slm_only
```

## 常用示例脚本

项目包含多个预配置的示例脚本,可以直接运行:

### 基线评估套件
```bash
bash examples/eval_baseline.sh
```
对比 SLM-only、Uncertainty routing 和 Speculative decoding 三种模式。

### 阈值扫描
```bash
bash examples/eval_threshold_sweep.sh
```
测试不同的 threshold 值,找到最优设置。

### 策略对比
```bash
bash examples/eval_compare_strategies.sh
```
对比三种策略 (speculative、uncertainty、entropy) 的性能。

### MMLU 评估
```bash
bash examples/eval_mmlu.sh
```
在 MMLU 基准上进行完整评估 (包含 5-shot)。

## 主要参数说明

### 模型配置
- `--slm`: 小模型 ID (必需)
- `--llm`: 大模型 ID (必需)
- `--device`: cuda 或 cpu
- `--dtype`: float16/bfloat16/float32

### 策略参数
- `--strategy`: speculative/uncertainty/entropy
- `--threshold`: 阈值 (uncertainty: 0.3-0.7, entropy: 1.5-3.0)
- `--num_drafts`: speculative 策略的草稿数量
- `--use_slm_only`: 仅使用 SLM (基线对比)

### 评估参数
- `--tasks`: 任务列表,用逗号分隔 (如 "hellaswag,arc_easy")
- `--num_fewshot`: few-shot 样本数量
- `--limit`: 限制测试样本数量 (用于快速测试)

## 分析结果

### 对比多个结果文件
```bash
python analyze_results.py compare \
    eval_results/baseline_slm_only.json \
    eval_results/baseline_uncertainty_t05.json
```

### 分析阈值扫描结果
```bash
python analyze_results.py threshold-sweep \
    eval_results/threshold_sweep/
```

### 生成结果摘要
```bash
python analyze_results.py summary eval_results/
```

## 常见任务

| 任务 | 描述 | 默认 few-shot |
|------|------|---------------|
| hellaswag | 常识推理 | 0 |
| arc_easy | 科学问题 (简单) | 0 |
| arc_challenge | 科学问题 (困难) | 0 |
| mmlu | 多任务语言理解 | 5 |
| gsm8k | 小学数学 | 5 |
| winogrande | 代词消歧 | 0 |

## 典型工作流程

### 1. 快速验证
```bash
# 先在小样本上测试
bash examples/eval_quick_test.sh
```

### 2. 基线评估
```bash
# 建立 SLM 基线
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --use_slm_only
```

### 3. 阈值调优
```bash
# 运行阈值扫描
bash examples/eval_threshold_sweep.sh

# 分析结果找到最优阈值
python analyze_results.py threshold-sweep eval_results/threshold_sweep/
```

### 4. 完整评估
```bash
# 使用最优阈值进行完整评估
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande \
    --strategy uncertainty \
    --threshold 0.5  # 使用第3步找到的最优值
```

### 5. 策略对比
```bash
# 对比不同策略
bash examples/eval_compare_strategies.sh
```

## 输出文件

评估结果保存为 JSON 格式:

```json
{
  "results": {
    "hellaswag": {
      "acc": 0.7234,
      "acc_norm": 0.7456,
      ...
    }
  },
  "metadata": {
    "slm_model_id": "Qwen/Qwen2.5-1.5B",
    "llm_model_id": "Qwen/Qwen2.5-7B",
    "strategy": "uncertainty",
    "threshold": 0.5,
    ...
  }
}
```

## 自定义模型

可以使用任何 HuggingFace 上的因果语言模型:

```bash
python eval_hybrid.py \
    --slm microsoft/phi-2 \
    --llm mistralai/Mistral-7B-v0.1 \
    --tasks hellaswag \
    --strategy uncertainty \
    --threshold 0.5
```

## 项目文件结构

```
hybrid_generator/
├── eval_hybrid.py              # 主评估脚本
├── analyze_results.py          # 结果分析工具
├── lm_eval_wrapper.py          # HybridLM 包装器
├── EVAL_README.md              # 详细文档
├── QUICKSTART.md               # 本文件
└── examples/                   # 示例脚本
    ├── eval_baseline.sh        # 基线评估
    ├── eval_threshold_sweep.sh # 阈值扫描
    ├── eval_quick_test.sh      # 快速测试
    ├── eval_mmlu.sh            # MMLU 评估
    └── eval_compare_strategies.sh # 策略对比
```

## 故障排除

### 内存不足 (OOM)
- 使用 `--dtype float16` 减少内存占用
- 使用更小的模型
- 确保 GPU 有足够内存同时加载两个模型

### 导入错误
确保在正确的目录运行:
```bash
cd /path/to/transformers/hybrid_generator
python eval_hybrid.py ...
```

### lm-eval 未安装
```bash
pip install lm-eval
```

## 更多帮助

- 查看详细文档: `cat EVAL_README.md`
- 查看所有可用参数: `python eval_hybrid.py --help`
- 列出所有任务: `python eval_hybrid.py --list_tasks`
- 查看配置但不运行: `python eval_hybrid.py --show_config ...`

## 推荐的评估顺序

1. **快速测试** → `bash examples/eval_quick_test.sh`
2. **SLM 基线** → 使用 `--use_slm_only`
3. **阈值扫描** → `bash examples/eval_threshold_sweep.sh`
4. **最优配置评估** → 使用找到的最优阈值
5. **策略对比** → `bash examples/eval_compare_strategies.sh`

祝评估顺利!
