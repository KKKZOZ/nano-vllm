#!/bin/bash
# Baseline evaluation: Compare SLM-only vs LLM-only vs HybridLM
# This script runs evaluations to establish baseline performance

set -e

# Configuration
SLM="Qwen/Qwen2.5-1.5B"
LLM="Qwen/Qwen2.5-7B"
TASKS="hellaswag,arc_easy,arc_challenge,winogrande"
DEVICE="cuda"
OUTPUT_DIR="./eval_results/baseline"

echo "========================================"
echo "Baseline Evaluation Suite"
echo "========================================"
echo "SLM: $SLM"
echo "LLM: $LLM"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Evaluate SLM only
echo "1/3 Evaluating SLM baseline..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --use_slm_only \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "baseline_slm_only.json"

echo ""
echo "SLM baseline completed!"
echo ""

# 2. Evaluate with uncertainty routing (threshold=0.5)
echo "2/3 Evaluating HybridLM with uncertainty routing..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy uncertainty \
    --threshold 0.5 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "baseline_uncertainty_t05.json"

echo ""
echo "Uncertainty routing completed!"
echo ""

# 3. Evaluate with speculative decoding
echo "3/3 Evaluating HybridLM with speculative decoding..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy speculative \
    --num_drafts 4 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "baseline_speculative_d4.json"

echo ""
echo "Speculative decoding completed!"
echo ""

echo "========================================"
echo "Baseline evaluation suite completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
