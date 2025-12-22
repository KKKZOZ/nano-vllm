#!/bin/bash
# Quick test: Fast evaluation on limited samples for testing
# Use this before running full evaluations to verify everything works

set -e

# Configuration
SLM="Qwen/Qwen2.5-1.5B"
LLM="Qwen/Qwen2.5-7B"
TASKS="hellaswag,arc_easy"
DEVICE="cuda"
LIMIT=50  # Only test on 50 examples per task

echo "========================================"
echo "Quick Test Evaluation"
echo "========================================"
echo "SLM: $SLM"
echo "LLM: $LLM"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT examples per task"
echo "========================================"
echo ""

echo "Testing uncertainty routing..."
python eval_hybrid.py \
    --slm "$SLM" \
    --llm "$LLM" \
    --tasks "$TASKS" \
    --strategy uncertainty \
    --threshold 0.5 \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --output_dir "./eval_results/quick_test" \
    --output_name "quick_test_uncertainty.json"

echo ""
echo "========================================"
echo "Quick test completed!"
echo "If this worked, you can run full evaluations."
echo "========================================"
