#!/bin/bash
# Threshold sweep: Test different thresholds to find optimal setting
# This helps determine the best threshold for uncertainty or entropy routing

set -e

# Configuration
SLM="Qwen/Qwen2.5-1.5B"
LLM="Qwen/Qwen2.5-7B"
TASKS="hellaswag"
STRATEGY="uncertainty"  # or "entropy"
DEVICE="cuda"
OUTPUT_DIR="./eval_results/threshold_sweep"

# Threshold values to test
THRESHOLDS=(0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)

echo "========================================"
echo "Threshold Sweep Evaluation"
echo "========================================"
echo "SLM: $SLM"
echo "LLM: $LLM"
echo "Tasks: $TASKS"
echo "Strategy: $STRATEGY"
echo "Thresholds: ${THRESHOLDS[@]}"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation for each threshold
total=${#THRESHOLDS[@]}
for i in "${!THRESHOLDS[@]}"; do
    threshold="${THRESHOLDS[$i]}"
    current=$((i + 1))

    echo "[$current/$total] Testing threshold: $threshold"

    python eval_hybrid.py \
        --slm "$SLM" \
        --llm "$LLM" \
        --tasks "$TASKS" \
        --strategy "$STRATEGY" \
        --threshold "$threshold" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_DIR" \
        --output_name "${STRATEGY}_threshold_${threshold}.json"

    echo ""
done

echo "========================================"
echo "Threshold sweep completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results, use:"
echo "python analyze_threshold_sweep.py $OUTPUT_DIR"
echo "========================================"
