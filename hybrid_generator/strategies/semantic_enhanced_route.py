import random
import time
from typing import Tuple, cast

import torch

from hybrid_generator.backends import BackendId, HybridBackend
from hybrid_generator.strategies.base import GenerationStrategy
from hybrid_generator.strategies.metrics import LiveMetricsTracker
from hybrid_generator.strategies.utils import (
    calculate_token_entropy,
    sample_token_flashinfer,
)
from nanovllm.utils.logger import logger


class SemanticEnhancedRouteStrategy(GenerationStrategy):
    """
    Entropy-based routing for the reasoning segment, SLM-only after </think>.

    This strategy requires a HybridBackend and uses the route logic only for
    tokens produced before the closing </think> tag.
    """

    def generate(
        self,
        hybrid_backend: HybridBackend,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        device: str,
        threshold: float = 2.0,
        llm_consecutive_tokens: int = 1,
        verbose: bool = False,
        report_live_metrics: bool = False,
        **kwargs,
    ) -> Tuple[str, dict]:
        """Generate using entropy-based routing before </think>, SLM-only after."""
        if not isinstance(hybrid_backend, HybridBackend):
            raise ValueError(
                "SemanticEnhancedRouteStrategy requires a HybridBackend instance."
            )

        # 1. Initialize Session
        req_id = random.randint(0, 2**31 - 1)

        # Initialize live metrics tracker
        live_tracker = (
            LiveMetricsTracker(report_interval=100) if report_live_metrics else None
        )

        # 2. Prefill Phase
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0].tolist()
        prompt_len = len(input_ids)

        # Prefill both models
        slm_logits_all = hybrid_backend.forward(BackendId.SLM, req_id, input_ids)
        current_slm_logits = slm_logits_all[-1, :]
        _ = hybrid_backend.forward(BackendId.LLM, req_id, input_ids)

        generated_ids = list(input_ids)

        slm_synced_len = prompt_len
        llm_synced_len = prompt_len

        eos_token_id = tokenizer.eos_token_id
        end_tag_ids = tokenizer.encode("</think>", add_special_tokens=False)
        reasoning_active = True
        logger.info(f"Using </think> tag IDs: {end_tag_ids}")

        enable_stats_sync = kwargs.get("enable_stats_sync", False)
        if enable_stats_sync and device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

        def _end_tag_emitted(last_token_id: int | None = None) -> bool:
            if not end_tag_ids:
                return False
            if len(end_tag_ids) == 1:
                if last_token_id is None:
                    return False
                return last_token_id == end_tag_ids[0]
            gen_len = len(generated_ids) - prompt_len
            if gen_len < len(end_tag_ids):
                return False
            start = len(generated_ids) - len(end_tag_ids)
            return generated_ids[start:] == end_tag_ids

        def _process_token(next_token_id, model_used, entropy_val, uncertainty_val):
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            if live_tracker:
                live_tracker.add_token(
                    model_used=model_used,
                    token_text=token_text,
                    entropy=entropy_val.item()
                    if hasattr(entropy_val, "item")
                    else entropy_val,
                    uncertainty=uncertainty_val.item()
                    if hasattr(uncertainty_val, "item")
                    else uncertainty_val,
                )
                if live_tracker.should_report():
                    live_tracker.print_progress()
            if verbose:
                print(f"{token_text}", end="", flush=True)
            return next_token_id == eos_token_id

        # 3. Main Generation Loop
        while len(generated_ids) - prompt_len < max_new_tokens:
            decode_steps += 1

            if reasoning_active:
                # --- Decision Phase ---
                entropy = calculate_token_entropy(current_slm_logits, temperature)
                use_llm = entropy >= threshold

                if not use_llm:
                    # === SLM Generation Branch ===
                    next_token_tensor = sample_token_flashinfer(
                        current_slm_logits.unsqueeze(0), temperature, top_k, top_p
                    )
                    next_token_id = cast(int, next_token_tensor.item())
                    slm_tokens += 1
                    generated_ids.append(next_token_id)

                    current_slm_logits = hybrid_backend.forward(
                        BackendId.SLM, req_id, [next_token_id]
                    )[-1, :]
                    slm_synced_len += 1

                    if _process_token(next_token_id, "slm", entropy, None):
                        break

                    if _end_tag_emitted(next_token_id):
                        logger.info(
                            "Detected </think> tag, switching to SLM-only mode."
                        )
                        reasoning_active = False
                else:
                    # === LLM Generation Branch ===
                    llm_next_logits = None
                    catchup_tokens = generated_ids[llm_synced_len:]

                    if catchup_tokens:
                        llm_next_logits = hybrid_backend.forward(
                            BackendId.LLM, req_id, catchup_tokens
                        )[-1, :]
                        llm_synced_len += len(catchup_tokens)

                    hit_eos = False
                    for _ in range(llm_consecutive_tokens):
                        if len(generated_ids) - prompt_len >= max_new_tokens:
                            break

                        if llm_next_logits is None:
                            llm_next_logits = hybrid_backend.forward(
                                BackendId.LLM, req_id, [generated_ids[-1]]
                            )[-1, :]
                            llm_synced_len += 1

                        next_token_tensor = sample_token_flashinfer(
                            llm_next_logits.unsqueeze(0), temperature, top_k, top_p
                        )
                        next_token_id = next_token_tensor.item()
                        llm_tokens += 1
                        generated_ids.append(next_token_id)

                        llm_next_logits = None

                        if _process_token(next_token_id, "llm", entropy, None):
                            hit_eos = True
                            break

                        if _end_tag_emitted(next_token_id):
                            logger.info(
                                "Detected </think> tag, switching to SLM-only mode."
                            )
                            reasoning_active = False
                            break

                    if hit_eos:
                        break

                # --- Synchronization Phase (SLM Catch-up) ---
                missing_tokens = generated_ids[slm_synced_len:]
                if missing_tokens:
                    slm_logits_all = hybrid_backend.forward(
                        BackendId.SLM, req_id, missing_tokens
                    )
                    current_slm_logits = slm_logits_all[-1, :]
                    slm_synced_len += len(missing_tokens)
            else:
                # === Non-reasoning: SLM-only ===
                next_token_tensor = sample_token_flashinfer(
                    current_slm_logits.unsqueeze(0), temperature, top_k, top_p
                )
                next_token_id = cast(int, next_token_tensor.item())
                slm_tokens += 1
                generated_ids.append(next_token_id)

                current_slm_logits = hybrid_backend.forward(
                    BackendId.SLM, req_id, [next_token_id]
                )[-1, :]
                slm_synced_len += 1

                if _process_token(next_token_id, "slm", 0.0, None):
                    break

        # 4. Cleanup
        hybrid_backend.free(BackendId.SLM, req_id)
        hybrid_backend.free(BackendId.LLM, req_id)

        if enable_stats_sync and device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        stats = {
            "total_tokens": len(generated_ids) - prompt_len,
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "elapsed_time": end_time - start_time,
            "threshold": threshold,
            "llm_consecutive_tokens": llm_consecutive_tokens,
        }

        return tokenizer.decode(
            generated_ids[prompt_len:], skip_special_tokens=True
        ), stats
