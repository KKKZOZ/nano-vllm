import argparse
import json
import os
from datetime import datetime
from typing import Any

import torch

from hybrid_generator import HybridGenerator

DEFAULT_STOP_STRINGS = ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]


def _parse_percentiles(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    percentiles = [int(p) for p in parts]
    for p in percentiles:
        if p <= 0 or p > 100:
            raise ValueError(f"Invalid percentile: {p}")
    return percentiles


def _parse_stop_strings(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    stop_strings = []
    for part in parts:
        try:
            part = bytes(part, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            pass
        stop_strings.append(part)
    return stop_strings or None


def _build_chat_messages(
    prompt: str, system_prompt: str | None
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


def _load_prompts(path: str, prompt_key: str, limit: int | None = None) -> list[str]:
    if limit is not None and limit <= 0:
        raise ValueError("Prompt limit must be a positive integer.")
    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict) and "problems" in data:
        items = data["problems"]
    else:
        items = data

    if not isinstance(items, list):
        raise ValueError("Expected a list of problem objects in the JSON file.")

    prompts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if prompt_key not in item:
            raise ValueError(f"Missing prompt key '{prompt_key}' in JSON item.")
        prompts.append(str(item[prompt_key]))
        if limit is not None and len(prompts) >= limit:
            break

    if not prompts:
        raise ValueError("No prompts were loaded from the JSON file.")

    return prompts


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _round_floats(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_floats(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    return value


def _make_output_dir(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_entropy_distribution(entropies: list[float], output_path: str) -> None:
    if not entropies:
        raise ValueError("No entropy values to plot.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save entropy distribution plots."
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropies, bins=50, color="#4c72b0", alpha=0.85)
    ax.set_title("Entropy Distribution")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Token Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate entropy/uncertainty distribution on AIME prompts."
    )
    parser.add_argument(
        "--input",
        default="aime_problems.json",
        help="Path to the AIME problems JSON file.",
    )
    parser.add_argument(
        "--prompt-key",
        default="problem",
        help="JSON field name to use as prompt text.",
    )
    parser.add_argument(
        "--slm-model-id",
        required=True,
        help="HuggingFace model ID for the SLM backend.",
    )
    parser.add_argument(
        "--llm-model-id",
        default=None,
        help="Optional HuggingFace model ID for the LLM backend.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Model dtype: float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max new tokens per prompt.",
    )
    parser.add_argument(
        "--stop-strings",
        default=",".join(DEFAULT_STOP_STRINGS),
        help=(
            "Comma-separated stop strings (supports \\n escapes). "
            "Default matches lm-eval aime24."
        ),
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        default=True,
        help="Wrap prompts with tokenizer.apply_chat_template().",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt used with --apply-chat-template.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--metric",
        choices=["entropy", "uncertainty"],
        default="entropy",
        help="Metric to aggregate across prompts.",
    )
    parser.add_argument(
        "--percentiles",
        default=None,
        help="Comma-separated percentiles, e.g. 10,20,30.",
    )
    parser.add_argument(
        "--prompt-limit",
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Limit number of prompts loaded from the input file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path to save calibration results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tokens during generation.",
    )
    parser.add_argument(
        "--report-live-metrics",
        action="store_true",
        help="Report live metrics every 100 tokens.",
    )
    parser.add_argument(
        "--enable-stats-sync",
        action="store_true",
        help="Synchronize CUDA stats for more accurate timing.",
    )

    args = parser.parse_args()

    prompts = _load_prompts(args.input, args.prompt_key, args.limit)

    generator = HybridGenerator(
        slm_model_id=args.slm_model_id,
        llm_model_id=args.llm_model_id,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
        verbose=args.verbose,
        report_live_metrics=args.report_live_metrics,
        enable_stats_sync=args.enable_stats_sync,
    )

    if args.apply_chat_template:
        tokenizer = generator.tokenizer
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template.")
        prompts = [
            tokenizer.apply_chat_template(
                _build_chat_messages(prompt, args.system_prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    print(
        f"temperature: {args.temperature}, top_k: {args.top_k}, top_p: {args.top_p}, min_p: {args.min_p}"
    )
    stop_strings = _parse_stop_strings(args.stop_strings)

    result = generator.generate_with_profile_calibration(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        stop_strings=stop_strings,
        metric=args.metric,
        percentiles=_parse_percentiles(args.percentiles),
        return_profiles=True,
    )

    profiles = result.pop("profiles", [])
    result = _round_floats(result, 4)

    output_dir = _make_output_dir("calibrate_result")
    calibration_path = os.path.join(output_dir, "calibration.json")
    inference_output_path = os.path.join(output_dir, "inference_outputs.json")
    entropy_plot_path = os.path.join(output_dir, "entropy_distribution.png")

    inference_outputs = []
    entropies: list[float] = []
    for idx, profile in enumerate(profiles):
        inference_outputs.append(
            {
                "index": idx,
                "prompt": prompts[idx],
                "generated_text": profile.generated_text,
                "total_tokens": len(profile.tokens),
                "total_time": profile.total_time,
            }
        )
        entropies.extend([float(token.entropy) for token in profile.tokens])

    inference_outputs = _round_floats(inference_outputs, 4)
    with open(inference_output_path, "w", encoding="utf-8") as f:
        json.dump(inference_outputs, f, ensure_ascii=False, indent=2)

    _save_entropy_distribution(entropies, entropy_plot_path)

    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved calibration result to: {calibration_path}")
    print(f"Saved inference outputs to: {inference_output_path}")
    print(f"Saved entropy distribution plot to: {entropy_plot_path}")

    if args.output and args.output != calibration_path:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved calibration result to: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
