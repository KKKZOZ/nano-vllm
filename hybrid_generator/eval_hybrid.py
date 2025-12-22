#!/usr/bin/env python3
"""
Evaluation script for HybridLM using lm-evaluation-harness.

This script provides a comprehensive interface to evaluate HybridLM on various
benchmarks with support for both common lm_eval parameters and HybridLM-specific
configuration options.

Usage examples:
    # Basic evaluation with uncertainty strategy
    python eval_hybrid.py \\
        --slm Qwen/Qwen3-1.7B \\
        --llm Qwen/Qwen3-8B \\
        --tasks hellaswag,arc_easy \\
        --strategy uncertainty \\
        --threshold 0.5

    # Baseline: evaluate SLM only
    python eval_hybrid.py \\
        --slm Qwen/Qwen3-1.7B \\
        --llm Qwen/Qwen3-8B \\
        --tasks mmlu \\
        --use_slm_only \\
        --num_fewshot 5

    # Speculative decoding with custom parameters
    python eval_hybrid.py \\
        --slm Qwen/Qwen3-1.7B \\
        --llm Qwen/Qwen3-8B \\
        --tasks gsm8k \\
        --strategy speculative \\
        --num_drafts 6 \\
        --limit 100
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the script's directory to Python path to allow imports
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.registry import MODEL_REGISTRY
except ImportError:
    print("ERROR: lm-evaluation-harness not installed.")
    print("Install with: pip install lm-eval")
    sys.exit(1)

# Import HybridLM
try:
    from lm_eval_wrapper import HybridLM
except ImportError:
    print("ERROR: Could not import HybridLM.")
    print(f"Script directory: {script_dir}")
    print(f"Python path: {sys.path[:3]}")
    print("\nMake sure lm_eval_wrapper.py exists in the hybrid_generator directory.")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate HybridLM using lm-evaluation-harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--slm",
        "--slm_model_id",
        dest="slm_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID for the small/fast model (SLM)",
    )
    model_group.add_argument(
        "--llm",
        "--llm_model_id",
        dest="llm_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID for the large/accurate model (LLM)",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )

    # HybridLM-specific arguments
    hybrid_group = parser.add_argument_group("HybridLM Strategy Parameters")
    hybrid_group.add_argument(
        "--strategy",
        type=str,
        default="uncertainty",
        choices=["speculative", "uncertainty", "entropy"],
        help=(
            "Generation strategy: "
            "'speculative' for standard speculative decoding, "
            "'uncertainty' for aleatoric uncertainty-based routing, "
            "'entropy' for entropy-based routing"
        ),
    )
    hybrid_group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for uncertainty/entropy routing strategies. "
            "When SLM's uncertainty/entropy exceeds this value, route to LLM. "
            "Typical range: 0.3-0.7 for uncertainty, 1.5-3.0 for entropy"
        ),
    )
    hybrid_group.add_argument(
        "--num_drafts",
        type=int,
        default=4,
        help="Number of draft tokens for speculative decoding strategy",
    )
    hybrid_group.add_argument(
        "--use_slm_only",
        action="store_true",
        help=(
            "Use only the SLM for all operations (baseline comparison) and skip loading the LLM."
        ),
    )
    hybrid_group.add_argument(
        "--report_routing_metrics",
        "--report-routing-metrics",
        dest="report_routing_metrics",
        action="store_true",
        help="Print and store token routing metrics (SLM vs LLM tokens) after evaluation",
    )
    hybrid_group.add_argument(
        "--report_content",
        "--report-content",
        dest="report_content",
        action="store_true",
        help="Record and report detailed content for each question (question text, answer, tokens generated, generation time)",
    )

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Parameters")
    eval_group.add_argument(
        "--tasks",
        type=str,
        required=True,
        help=(
            "Comma-separated list of tasks to evaluate on. "
            "Examples: 'hellaswag,arc_easy', 'mmlu', 'gsm8k'. "
            "Use 'lm_eval --tasks list' to see all available tasks"
        ),
    )
    eval_group.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help=(
            "Number of few-shot examples to use for evaluation. "
            "If not specified, uses the task's default. Common values: 0, 5"
        ),
    )
    eval_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (currently only batch_size=1 is supported)",
    )
    eval_group.add_argument(
        "--limit",
        type=float,
        default=None,
        help=(
            "Limit the number of examples per task. "
            "Can be an integer (number of examples) or float between 0-1 (fraction). "
            "Useful for quick testing. Example: --limit 100 or --limit 0.1"
        ),
    )
    eval_group.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=(
            "Additional generation kwargs as JSON string. "
            'Example: --gen_kwargs \'{"temperature": 0.8, "max_gen_toks": 512}\''
        ),
    )
    eval_group.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template to prompts before evaluation",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    output_group.add_argument(
        "--output_name",
        type=str,
        default=None,
        help=(
            "Custom name for the output file. "
            "If not specified, generates a name based on timestamp and configuration"
        ),
    )
    output_group.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual sample results to output file",
    )

    # Advanced arguments
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    advanced_group.add_argument(
        "--show_config",
        action="store_true",
        help="Show configuration and exit without running evaluation",
    )
    advanced_group.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available tasks and exit",
    )

    args = parser.parse_args()

    # Validation
    if args.batch_size != 1:
        print(
            "WARNING: HybridLM currently only supports batch_size=1. "
            "Setting batch_size=1."
        )
        args.batch_size = 1

    return args


def list_available_tasks():
    """Print all available tasks."""
    print("\n=== Available Tasks ===")
    print("\nCommon benchmarks:")
    common_tasks = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "mmlu",
        "truthfulqa_mc",
        "gsm8k",
        "winogrande",
        "piqa",
        "boolq",
        "openbookqa",
    ]
    for task in common_tasks:
        print(f"  - {task}")

    print("\nFor a complete list, run: lm_eval --tasks list")
    print()


def create_output_filename(args) -> str:
    """Generate output filename based on configuration."""
    if args.output_name:
        return args.output_name

    # Create descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slm_name = args.slm_model_id.split("/")[-1]
    llm_name = args.llm_model_id.split("/")[-1]
    tasks_str = args.tasks.replace(",", "_")

    if args.use_slm_only:
        mode = "slm_only"
    else:
        mode = f"{args.strategy}"
        if args.strategy in ["uncertainty", "entropy"]:
            mode += f"_t{args.threshold}"
        elif args.strategy == "speculative":
            mode += f"_d{args.num_drafts}"

    filename = f"hybrid_{slm_name}_vs_{llm_name}_{mode}_{tasks_str}_{timestamp}.json"
    return filename


def print_config(args):
    """Print evaluation configuration."""
    print("\n" + "=" * 70)
    print("HybridLM Evaluation Configuration")
    print("=" * 70)

    print("\nModel Configuration:")
    print(f"  SLM: {args.slm_model_id}")
    print(f"  LLM: {args.llm_model_id}")
    print(f"  Device: {args.device}")
    print(f"  Data type: {args.dtype}")

    print("\nStrategy Configuration:")
    if args.use_slm_only:
        print("  Mode: SLM Only (baseline)")
    else:
        print(f"  Strategy: {args.strategy}")
        if args.strategy in ["uncertainty", "entropy"]:
            print(f"  Threshold: {args.threshold}")
        elif args.strategy == "speculative":
            print(f"  Number of drafts: {args.num_drafts}")

    print("\nEvaluation Configuration:")
    print(f"  Tasks: {args.tasks}")
    print(
        f"  Few-shot examples: {args.num_fewshot if args.num_fewshot is not None else 'default'}"
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Limit: {args.limit if args.limit is not None else 'none'}")
    print(f"  Apply chat template: {args.apply_chat_template}")
    if args.gen_kwargs:
        print(f"  Generation kwargs: {args.gen_kwargs}")

    print("\nOutput Configuration:")
    print(f"  Output directory: {args.output_dir}")
    output_file = create_output_filename(args)
    print(f"  Output file: {output_file}")
    print(f"  Log samples: {args.log_samples}")
    print(f"  Report routing metrics: {args.report_routing_metrics}")
    print(f"  Report content: {args.report_content}")

    print("=" * 70 + "\n")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Handle special commands
    if args.list_tasks:
        list_available_tasks()
        return

    if args.show_config:
        print_config(args)
        return

    # Print configuration
    print_config(args)

    # Parse tasks
    task_list = [task.strip() for task in args.tasks.split(",")]

    # Parse generation kwargs if provided
    gen_kwargs = {}
    if args.gen_kwargs:
        try:
            gen_kwargs = json.loads(args.gen_kwargs)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --gen_kwargs: {e}")
            sys.exit(1)

    # Initialize HybridLM
    print("Initializing HybridLM...")
    try:
        model = HybridLM(
            slm_model_id=args.slm_model_id,
            llm_model_id=args.llm_model_id,
            strategy=args.strategy,
            threshold=args.threshold,
            num_drafts=args.num_drafts,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            use_slm_only=args.use_slm_only,
            report_routing_metrics=args.report_routing_metrics,
            report_content=args.report_content,
        )
        print(f"Successfully initialized: {model}")
    except Exception as e:
        print(f"ERROR: Failed to initialize HybridLM: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run evaluation
    print("\nStarting evaluation...")
    print("-" * 70)

    try:
        results = evaluator.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=args.log_samples,
            verbosity=args.verbosity,
            apply_chat_template=args.apply_chat_template,
            **gen_kwargs,
        )
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save results
    print("\n" + "-" * 70)
    print("Evaluation completed!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / create_output_filename(args)

    # Add metadata to results
    results["metadata"] = {
        "slm_model_id": args.slm_model_id,
        "llm_model_id": args.llm_model_id,
        "strategy": args.strategy,
        "threshold": args.threshold,
        "num_drafts": args.num_drafts,
        "use_slm_only": args.use_slm_only,
        "device": args.device,
        "dtype": args.dtype,
        "tasks": task_list,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "apply_chat_template": args.apply_chat_template,
        "timestamp": datetime.now().isoformat(),
        "report_routing_metrics": args.report_routing_metrics,
        "report_content": args.report_content,
    }

    if args.report_routing_metrics:
        results["metadata"]["routing_metrics"] = model.get_routing_metrics()

    if args.report_content:
        results["metadata"]["content_metrics"] = model.get_content_metrics()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("=" * 70)

    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\nTask: {task_name}")
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")

    if args.report_routing_metrics:
        model.report_routing_metrics()

    if args.report_content:
        model.report_content()

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
