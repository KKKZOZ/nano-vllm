#!/bin/bash
# MMLU evaluation: Comprehensive evaluation on MMLU benchmark
# MMLU is a challenging multi-task benchmark with 5-shot prompting

set -e

# Configuration
SLM="Qwen/Qwen2.5-1.5B"
LLM="Qwen/Qwen2.5-7B"
TASKS="mmlu"
NUM_FEWSHOT=5  # MMLU standard is 5-shot
DEVICE="cuda"
OUTPUT_DIR="./eval_results/mmlu"

echo "========================================"
echo "MMLU Evaluation"
echo "========================================"
echo "SLM: $SLM"
echo "LLM: $LLM"
echo "Tasks: $TASKS"
echo "Few-shot: $NUM_FEWSHOT"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. SLM baseline
echo "1/3 Evaluating SLM baseline..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --use_slm_only \
    --num_fewshot "$NUM_FEWSHOT" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "mmlu_slm_only.json" \
    --log_samples

echo ""

# 2. Uncertainty routing
echo "2/3 Evaluating uncertainty routing..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy uncertainty \
    --threshold 0.5 \
    --num_fewshot "$NUM_FEWSHOT" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "mmlu_uncertainty_t05.json" \
    --log_samples

echo ""

# 3. Speculative decoding
echo "3/3 Evaluating speculative decoding..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy speculative \
    --num_drafts 4 \
    --num_fewshot "$NUM_FEWSHOT" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "mmlu_speculative_d4.json" \
    --log_samples

echo ""
echo "========================================"
echo "MMLU evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
