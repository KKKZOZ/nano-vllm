#!/bin/bash
# Compare all three strategies on the same tasks
# This helps determine which strategy works best for your use case

set -e

# Configuration
SLM="Qwen/Qwen2.5-1.5B"
LLM="Qwen/Qwen2.5-7B"
TASKS="hellaswag,arc_easy,arc_challenge"
DEVICE="cuda"
OUTPUT_DIR="./eval_results/strategy_comparison"

echo "========================================"
echo "Strategy Comparison Evaluation"
echo "========================================"
echo "SLM: $SLM"
echo "LLM: $LLM"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Speculative decoding
echo "1/3 Evaluating Speculative Decoding..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy speculative \
    --num_drafts 4 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "strategy_speculative.json"

echo ""

# 2. Uncertainty routing
echo "2/3 Evaluating Uncertainty Routing..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy uncertainty \
    --threshold 0.5 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "strategy_uncertainty.json"

echo ""

# 3. Entropy routing
echo "3/3 Evaluating Entropy Routing..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy entropy \
    --threshold 2.0 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "strategy_entropy.json"

echo ""
echo "========================================"
echo "Strategy comparison completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Compare results to see which strategy works best."
echo "========================================"
