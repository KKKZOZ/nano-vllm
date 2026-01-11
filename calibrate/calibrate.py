import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import torch

from hybrid_generator import HybridGenerator

DEFAULT_STOP_STRINGS = ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]
DEFAULT_PROMPT_KEYS = ["problem", "prompt", "question"]
DEFAULT_CONTAINER_KEYS = ["problems", "data", "examples", "questions", "items"]


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
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = "You are a helpful assistant."
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _parse_prompt_keys(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def _read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_file(path: str) -> list[Any]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_data(path: str, input_format: str) -> Any:
    if input_format not in {"auto", "json", "jsonl"}:
        raise ValueError(f"Unsupported input format: {input_format}")
    if input_format == "json":
        return _read_json_file(path)
    if input_format == "jsonl":
        return _read_jsonl_file(path)
    if path.endswith((".jsonl", ".jsonlines")):
        return _read_jsonl_file(path)
    try:
        return _read_json_file(path)
    except json.JSONDecodeError:
        return _read_jsonl_file(path)


def _get_by_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_items(data: Any, data_key: str | None) -> list[Any]:
    if data_key:
        items = _get_by_path(data, data_key)
        if items is None:
            raise ValueError(f"Data key '{data_key}' not found in JSON input.")
        if not isinstance(items, list):
            raise ValueError(f"Data key '{data_key}' must point to a list.")
        return items

    if isinstance(data, dict):
        for key in DEFAULT_CONTAINER_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
        list_keys = [key for key, value in data.items() if isinstance(value, list)]
        if len(list_keys) == 1:
            return data[list_keys[0]]
        if list_keys:
            raise ValueError(
                "Multiple list fields found in JSON input. "
                "Use --data-key to specify which list to read."
            )
        raise ValueError(
            "No list field found in JSON input. Use --data-key to specify a list."
        )

    if isinstance(data, list):
        return data

    raise ValueError("Expected a list of prompts in the JSON input.")


def _extract_prompt(item: Any, prompt_keys: list[str]) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    for key in prompt_keys:
        value = _get_by_path(item, key)
        if value is not None:
            return str(value)
    return None


def _load_prompts(
    path: str,
    prompt_key: str | None,
    data_key: str | None,
    input_format: str,
    limit: int | None = None,
) -> list[str]:
    if limit is not None and limit <= 0:
        raise ValueError("Prompt limit must be a positive integer.")
    data = _load_data(path, input_format)
    items = _resolve_items(data, data_key)
    prompt_keys = _parse_prompt_keys(prompt_key) or DEFAULT_PROMPT_KEYS

    prompts = []
    for item in items:
        prompt = _extract_prompt(item, prompt_keys)
        if prompt is None:
            raise ValueError(
                f"Missing prompt key(s) {prompt_keys} in JSON item."
            )
        prompts.append(prompt)
        if limit is not None and len(prompts) >= limit:
            break

    if not prompts:
        raise ValueError("No prompts were loaded from the input file.")

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


def _highlight_token(token_text: str) -> str:
    if not token_text:
        return "@@"
    leading_len = len(token_text) - len(token_text.lstrip())
    trailing_len = len(token_text) - len(token_text.rstrip())
    core_start = leading_len
    core_end = len(token_text) - trailing_len
    if core_end <= core_start:
        return f"@{token_text}@"
    leading = token_text[:core_start]
    core = token_text[core_start:core_end]
    trailing = token_text[core_end:]
    return f"{leading}@{core}@{trailing}"


def _token_label(token_text: str) -> str:
    if not token_text:
        return ""
    stripped = token_text.strip()
    return stripped if stripped else token_text


def _format_entropy(value: float) -> str:
    return f"{value:.4f}"


def _single_line_text(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _build_high_entropy_contexts(
    profiles: list[Any], top_percent: int = 20, window: int = 15
) -> list[str]:
    candidates: list[tuple[float, int, int]] = []
    for profile_idx, profile in enumerate(profiles):
        for token_idx, token in enumerate(profile.tokens):
            candidates.append((float(token.entropy), profile_idx, token_idx))

    if not candidates:
        return []

    top_count = int(math.ceil(len(candidates) * top_percent / 100))
    top_count = max(1, min(len(candidates), top_count))
    candidates.sort(key=lambda item: item[0], reverse=True)

    contexts: list[str] = []
    for entropy, profile_idx, token_idx in candidates[:top_count]:
        profile = profiles[profile_idx]
        start = max(0, token_idx - window)
        end = min(len(profile.tokens), token_idx + window + 1)
        before_text = "".join(
            token.token_text for token in profile.tokens[start:token_idx]
        )
        token_text = profile.tokens[token_idx].token_text
        highlighted = _highlight_token(token_text)
        after_text = "".join(
            token.token_text for token in profile.tokens[token_idx + 1 : end]
        )
        label = _token_label(token_text)
        prefix = f"[{label}: {_format_entropy(entropy)}] " if label else ""
        contexts.append(
            _single_line_text(f"{prefix}{before_text}{highlighted}{after_text}")
        )
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate entropy/uncertainty distribution on dataset prompts."
    )
    parser.add_argument(
        "--input",
        default="aime_problems.json",
        help="Path to the dataset JSON/JSONL file.",
    )
    parser.add_argument(
        "--prompt-key",
        default=",".join(DEFAULT_PROMPT_KEYS),
        help=(
            "Comma-separated JSON field names to use as prompt text "
            "(supports dot paths like data.question)."
        ),
    )
    parser.add_argument(
        "--data-key",
        default=None,
        help=(
            "Optional JSON field path that contains the list of items "
            "(supports dot paths like data.examples)."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Input format for the dataset file.",
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
    parser.add_argument(
        "--details",
        action="store_true",
        help="Save per-token entropy details and high-entropy context snippets.",
    )

    args = parser.parse_args()

    prompts = _load_prompts(
        args.input, args.prompt_key, args.data_key, args.input_format, args.limit
    )

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
    token_entropy_path = os.path.join(output_dir, "token_entropies.json")
    high_entropy_contexts_path = os.path.join(
        output_dir, "high_entropy_contexts.txt"
    )

    inference_outputs = []
    token_entropy_outputs = []
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
        if args.details:
            token_entropy_outputs.append(
                {
                    "index": idx,
                    "tokens": [
                        {
                            "position": token.position,
                            "token": token.token_text,
                            "entropy": float(token.entropy),
                        }
                        for token in profile.tokens
                    ],
                }
            )

    inference_outputs = _round_floats(inference_outputs, 4)
    with open(inference_output_path, "w", encoding="utf-8") as f:
        json.dump(inference_outputs, f, ensure_ascii=False, indent=2)

    _save_entropy_distribution(entropies, entropy_plot_path)

    if args.details:
        token_entropy_outputs = _round_floats(token_entropy_outputs, 4)
        with open(token_entropy_path, "w", encoding="utf-8") as f:
            json.dump(token_entropy_outputs, f, ensure_ascii=False, indent=2)
        high_entropy_contexts = _build_high_entropy_contexts(
            profiles, top_percent=20, window=15
        )
        with open(high_entropy_contexts_path, "w", encoding="utf-8") as f:
            for line in high_entropy_contexts:
                f.write(f"{line}\n")

    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved calibration result to: {calibration_path}")
    print(f"Saved inference outputs to: {inference_output_path}")
    print(f"Saved entropy distribution plot to: {entropy_plot_path}")
    if args.details:
        print(f"Saved token entropies to: {token_entropy_path}")
        print(f"Saved high-entropy contexts to: {high_entropy_contexts_path}")

    if args.output and args.output != calibration_path:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved calibration result to: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
