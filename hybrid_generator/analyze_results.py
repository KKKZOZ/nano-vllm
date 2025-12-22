#!/usr/bin/env python3
"""
Analyze and compare HybridLM evaluation results.

This script provides utilities to:
1. Compare results from different configurations
2. Analyze threshold sweep results
3. Generate summary tables and plots

Usage:
    # Compare multiple result files
    python analyze_results.py compare result1.json result2.json result3.json

    # Analyze threshold sweep
    python analyze_results.py threshold-sweep ./eval_results/threshold_sweep/

    # Generate summary table
    python analyze_results.py summary ./eval_results/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_result(path: Path) -> Dict[str, Any]:
    """Load a result JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from result."""
    metrics = {}
    if "results" in result:
        for task_name, task_results in result["results"].items():
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}_{metric_name}"] = value
    return metrics


def compare_results(result_files: List[Path]):
    """Compare results from multiple files."""
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("=" * 80 + "\n")

    results = []
    for path in result_files:
        try:
            result = load_result(path)
            results.append((path.name, result))
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not results:
        print("No valid result files found.")
        return

    # Extract all tasks and metrics
    all_tasks = set()
    for _, result in results:
        if "results" in result:
            all_tasks.update(result["results"].keys())

    # Print metadata comparison
    print("Configuration Comparison:")
    print("-" * 80)
    headers = ["File", "SLM", "LLM", "Strategy", "Threshold", "Mode"]
    print(
        f"{headers[0]:30s} {headers[1]:15s} {headers[2]:15s} {headers[3]:12s} {headers[4]:10s} {headers[5]:10s}"
    )
    print("-" * 80)

    for filename, result in results:
        metadata = result.get("metadata", {})
        slm = metadata.get("slm_model_id", "N/A").split("/")[-1][:15]
        llm = metadata.get("llm_model_id", "N/A").split("/")[-1][:15]
        strategy = metadata.get("strategy", "N/A")
        threshold = metadata.get("threshold", "N/A")
        mode = "SLM-only" if metadata.get("use_slm_only", False) else "Hybrid"

        print(
            f"{filename:30s} {slm:15s} {llm:15s} {strategy:12s} {threshold!s:10s} {mode:10s}"
        )

    print("\n")

    # Print metrics comparison for each task
    for task in sorted(all_tasks):
        print(f"Task: {task}")
        print("-" * 80)

        # Get all metrics for this task
        task_metrics = set()
        for _, result in results:
            if "results" in result and task in result["results"]:
                task_metrics.update(result["results"][task].keys())

        # Filter to numeric metrics
        numeric_metrics = []
        for metric in task_metrics:
            sample_result = results[0][1]["results"].get(task, {}).get(metric)
            if isinstance(sample_result, (int, float)):
                numeric_metrics.append(metric)

        # Print each metric
        for metric in sorted(numeric_metrics):
            print(f"\n  {metric}:")
            for filename, result in results:
                value = result.get("results", {}).get(task, {}).get(metric, "N/A")
                if isinstance(value, float):
                    print(f"    {filename:30s}: {value:.4f}")
                else:
                    print(f"    {filename:30s}: {value}")

        print("\n")


def analyze_threshold_sweep(directory: Path):
    """Analyze results from a threshold sweep."""
    print("\n" + "=" * 80)
    print("Threshold Sweep Analysis")
    print("=" * 80 + "\n")

    # Find all result files
    result_files = sorted(directory.glob("*.json"))

    if not result_files:
        print(f"No result files found in {directory}")
        return

    # Load results and extract thresholds
    sweep_data = []
    for path in result_files:
        try:
            result = load_result(path)
            metadata = result.get("metadata", {})
            threshold = metadata.get("threshold")
            strategy = metadata.get("strategy")

            if threshold is not None:
                sweep_data.append((threshold, strategy, result))
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not sweep_data:
        print("No valid threshold sweep data found.")
        return

    # Sort by threshold
    sweep_data.sort(key=lambda x: x[0])

    # Print table
    print(f"Strategy: {sweep_data[0][1]}")
    print("-" * 80)

    # Get task name (assuming single task)
    first_result = sweep_data[0][2]
    tasks = list(first_result.get("results", {}).keys())

    if not tasks:
        print("No task results found.")
        return

    for task in tasks:
        print(f"\nTask: {task}")
        print("-" * 80)

        # Get metrics
        metrics = set()
        for _, _, result in sweep_data:
            task_results = result.get("results", {}).get(task, {})
            metrics.update(
                k for k, v in task_results.items() if isinstance(v, (int, float))
            )

        # Print header
        print(f"{'Threshold':>10s}", end="")
        for metric in sorted(metrics):
            print(f"  {metric:>12s}", end="")
        print()
        print("-" * 80)

        # Print data
        for threshold, _, result in sweep_data:
            task_results = result.get("results", {}).get(task, {})
            print(f"{threshold:>10.2f}", end="")
            for metric in sorted(metrics):
                value = task_results.get(metric, 0)
                if isinstance(value, float):
                    print(f"  {value:>12.4f}", end="")
                else:
                    print(f"  {value:>12}", end="")
            print()

        # Find optimal threshold based on primary metric
        primary_metric = "acc" if "acc" in metrics else sorted(metrics)[0]
        best_threshold = max(
            sweep_data,
            key=lambda x: x[2].get("results", {}).get(task, {}).get(primary_metric, 0),
        )[0]

        print(f"\nOptimal threshold (by {primary_metric}): {best_threshold:.2f}")

    print("\n")

    # Try to create a plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
        if len(tasks) == 1:
            axes = [axes]

        for idx, task in enumerate(tasks):
            ax = axes[idx]

            # Get metrics
            task_results = sweep_data[0][2].get("results", {}).get(task, {})
            metrics = [
                k for k, v in task_results.items() if isinstance(v, (int, float))
            ]

            # Plot each metric
            for metric in sorted(metrics):
                thresholds = [t for t, _, _ in sweep_data]
                values = [
                    r.get("results", {}).get(task, {}).get(metric, 0)
                    for _, _, r in sweep_data
                ]
                ax.plot(thresholds, values, marker="o", label=metric)

            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_title(f"Task: {task}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = directory / "threshold_sweep_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}\n")

    except ImportError:
        print("Note: Install matplotlib to generate plots: pip install matplotlib\n")


def generate_summary(directory: Path):
    """Generate summary table of all results in a directory."""
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80 + "\n")

    result_files = sorted(directory.glob("**/*.json"))

    if not result_files:
        print(f"No result files found in {directory}")
        return

    print(f"Found {len(result_files)} result file(s)\n")

    # Create summary table
    summaries = []
    for path in result_files:
        try:
            result = load_result(path)
            metadata = result.get("metadata", {})

            summary = {
                "file": path.relative_to(directory),
                "slm": metadata.get("slm_model_id", "N/A").split("/")[-1],
                "llm": metadata.get("llm_model_id", "N/A").split("/")[-1],
                "strategy": metadata.get("strategy", "N/A"),
                "threshold": metadata.get("threshold", "N/A"),
                "tasks": ",".join(metadata.get("tasks", [])),
            }

            # Add task results
            if "results" in result:
                for task_name, task_results in result["results"].items():
                    for metric_name, value in task_results.items():
                        if isinstance(value, (int, float)):
                            summary[f"{task_name}_{metric_name}"] = value

            summaries.append(summary)

        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not summaries:
        print("No valid results found.")
        return

    # Print table
    print(f"{'File':40s} {'Strategy':12s} {'Threshold':10s} {'Tasks':20s}")
    print("-" * 80)
    for summary in summaries:
        file_str = str(summary["file"])[:40]
        strategy = summary["strategy"][:12]
        threshold = str(summary["threshold"])[:10]
        tasks = summary["tasks"][:20]
        print(f"{file_str:40s} {strategy:12s} {threshold:10s} {tasks:20s}")

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HybridLM evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare results from multiple files"
    )
    compare_parser.add_argument(
        "files", nargs="+", type=Path, help="Result JSON files to compare"
    )

    # Threshold sweep command
    sweep_parser = subparsers.add_parser(
        "threshold-sweep", help="Analyze threshold sweep results"
    )
    sweep_parser.add_argument(
        "directory", type=Path, help="Directory containing threshold sweep results"
    )

    # Summary command
    summary_parser = subparsers.add_parser(
        "summary", help="Generate summary of all results in directory"
    )
    summary_parser.add_argument(
        "directory", type=Path, help="Directory containing result files"
    )

    args = parser.parse_args()

    if args.command == "compare":
        compare_results(args.files)
    elif args.command == "threshold-sweep":
        analyze_threshold_sweep(args.directory)
    elif args.command == "summary":
        generate_summary(args.directory)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
