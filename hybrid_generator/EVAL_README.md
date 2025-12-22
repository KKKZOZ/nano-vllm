# HybridLM Evaluation Guide

This guide explains how to evaluate HybridLM using the `eval_hybrid.py` script with lm-evaluation-harness.

## Prerequisites

```bash
# Install lm-evaluation-harness
pip install lm-eval

# Install required dependencies
pip install torch transformers
```

## Quick Start

### 1. Basic Evaluation

Evaluate HybridLM with uncertainty-based routing on HellaSwag:

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --strategy uncertainty \
    --threshold 0.5
```

### 2. Baseline Evaluation (SLM Only)

Evaluate only the SLM for baseline comparison:

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --use_slm_only
```

### 3. Multiple Tasks with Few-Shot

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag,arc_easy,arc_challenge \
    --strategy uncertainty \
    --threshold 0.5 \
    --num_fewshot 5
```

## Command-Line Arguments

### Model Configuration

- `--slm MODEL_ID`: HuggingFace model ID for small/fast model (required)
- `--llm MODEL_ID`: HuggingFace model ID for large/accurate model (required)
- `--device {cuda,cpu}`: Device to run on (default: cuda)
- `--dtype {float16,bfloat16,float32}`: Model data type (default: float16)

### HybridLM Strategy Parameters

- `--strategy {speculative,uncertainty,entropy}`: Generation strategy (default: uncertainty)
  - `speculative`: Standard speculative decoding with draft tokens
  - `uncertainty`: Route to LLM when aleatoric uncertainty is high
  - `entropy`: Route to LLM when entropy is high

- `--threshold FLOAT`: Threshold for routing (default: 0.5)
  - For `uncertainty`: typical range 0.3-0.7
  - For `entropy`: typical range 1.5-3.0
  - Higher threshold = more SLM usage, lower accuracy
  - Lower threshold = more LLM usage, higher accuracy

- `--num_drafts INT`: Number of draft tokens for speculative strategy (default: 4)

- `--use_slm_only`: Use only SLM (for baseline comparison)

### Evaluation Parameters

- `--tasks TASK1,TASK2,...`: Comma-separated list of tasks (required)
  - Common tasks: `hellaswag`, `arc_easy`, `arc_challenge`, `mmlu`, `gsm8k`, `winogrande`, `truthfulqa_mc`
  - Use `--list_tasks` to see available tasks

- `--num_fewshot INT`: Number of few-shot examples (default: task-specific)
  - Common values: 0 (zero-shot), 5 (5-shot)

- `--batch_size INT`: Batch size (default: 1, currently only 1 is supported)

- `--limit INT or FLOAT`: Limit number of test examples
  - Integer: exact number of examples (e.g., `--limit 100`)
  - Float 0-1: fraction of dataset (e.g., `--limit 0.1`)

- `--gen_kwargs JSON`: Additional generation parameters as JSON
  ```bash
  --gen_kwargs '{"temperature": 0.8, "max_gen_toks": 512}'
  ```

### Output Configuration

- `--output_dir PATH`: Directory to save results (default: ./eval_results)
- `--output_name NAME`: Custom output filename
- `--log_samples`: Save individual sample results

### Advanced Options

- `--verbosity {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)
- `--show_config`: Display configuration without running evaluation
- `--list_tasks`: List all available tasks

## Usage Examples

### Example 1: Compare Different Strategies

```bash
# Uncertainty-based routing
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --strategy uncertainty \
    --threshold 0.5

# Entropy-based routing
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --strategy entropy \
    --threshold 2.0

# Speculative decoding
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag \
    --strategy speculative \
    --num_drafts 4
```

### Example 2: Threshold Sweep

Test different uncertainty thresholds to find optimal setting:

```bash
for threshold in 0.3 0.4 0.5 0.6 0.7; do
    python eval_hybrid.py \
        --slm Qwen/Qwen2.5-1.5B \
        --llm Qwen/Qwen2.5-7B \
        --tasks hellaswag \
        --strategy uncertainty \
        --threshold $threshold \
        --output_name "uncertainty_t${threshold}.json"
done
```

### Example 3: Quick Test with Limited Samples

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks hellaswag,arc_easy \
    --strategy uncertainty \
    --threshold 0.5 \
    --limit 100  # Only test on 100 examples per task
```

### Example 4: Comprehensive MMLU Evaluation

```bash
python eval_hybrid.py \
    --slm Qwen/Qwen2.5-1.5B \
    --llm Qwen/Qwen2.5-7B \
    --tasks mmlu \
    --strategy uncertainty \
    --threshold 0.5 \
    --num_fewshot 5 \
    --output_name "mmlu_5shot_uncertainty.json" \
    --log_samples
```

### Example 5: Custom Model Configuration

```bash
# Use different models
python eval_hybrid.py \
    --slm microsoft/phi-2 \
    --llm mistralai/Mistral-7B-v0.1 \
    --tasks hellaswag \
    --strategy uncertainty \
    --threshold 0.5 \
    --dtype bfloat16
```

## Understanding Results

The script saves results to JSON files with the following structure:

```json
{
  "results": {
    "task_name": {
      "acc": 0.7234,
      "acc_norm": 0.7456,
      ...
    }
  },
  "metadata": {
    "slm_model_id": "...",
    "llm_model_id": "...",
    "strategy": "uncertainty",
    "threshold": 0.5,
    ...
  }
}
```

Key metrics:
- `acc`: Accuracy
- `acc_norm`: Normalized accuracy (for multiple-choice tasks)
- Task-specific metrics vary by benchmark

## Choosing the Right Threshold

### For Uncertainty Strategy

1. First, profile your SLM to understand its uncertainty distribution:
   ```python
   from hybrid_generator import HybridGenerator

   generator = HybridGenerator(
       slm_model_id="Qwen/Qwen2.5-1.5B",
       llm_model_id="Qwen/Qwen2.5-7B"
   )

   profile = generator.generate_with_profile(
       prompt="Your test prompt",
       max_new_tokens=500,
       enable_routing=False  # SLM only
   )

   profile.print_summary()
   ```

2. Look at the uncertainty distribution percentiles
3. Set threshold based on desired SLM usage:
   - Top 20-30% highest uncertainty → threshold ~0.5-0.6
   - Top 40-50% highest uncertainty → threshold ~0.4-0.5

### For Entropy Strategy

Similar process, but entropy values are typically higher:
- Typical range: 1.5 to 3.0
- Higher entropy indicates more uncertainty in the distribution

## Tips

1. **Start with a baseline**: Always run `--use_slm_only` first to establish baseline accuracy
2. **Use --limit for testing**: Test with `--limit 100` before running full evaluation
3. **Monitor GPU memory**: Both models are loaded in memory simultaneously
4. **Compare strategies**: Try all three strategies to find which works best for your use case
5. **Threshold tuning**: Run a sweep to find optimal threshold for your models and tasks

## Common Tasks

| Task | Description | Default Few-Shot |
|------|-------------|------------------|
| hellaswag | Commonsense reasoning | 0 |
| arc_easy | Science questions (easy) | 0 |
| arc_challenge | Science questions (hard) | 0 |
| mmlu | Massive multitask language understanding | 5 |
| gsm8k | Grade school math | 5 |
| winogrande | Pronoun resolution | 0 |
| truthfulqa_mc | Truthfulness in QA | 0 |
| piqa | Physical commonsense | 0 |
| boolq | Boolean questions | 0 |

## Troubleshooting

### Out of Memory
- Reduce model size or use `--dtype float16`
- Use smaller models
- Ensure GPU has sufficient memory for both models

### Import Errors
```bash
# Ensure lm_eval_wrapper is in the correct location
cd /path/to/transformers/hybrid_generator
python eval_hybrid.py ...
```

### Batch Size Warning
HybridLM currently only supports `batch_size=1`. This is automatically enforced.

## Next Steps

- See `examples/` directory for more example scripts
- Check `EVAL_EXAMPLES.md` for complete evaluation workflows
- Read the paper (if available) for optimal hyperparameter settings
