import random
import time
from typing import Tuple, cast

import torch
import torch.nn.functional as F

from hybrid_generator.backends import BackendId, HybridBackend
from hybrid_generator.strategies.base import GenerationStrategy
from hybrid_generator.strategies.metrics import LiveMetricsTracker
from hybrid_generator.strategies.utils import (
    calculate_token_entropy,
    sample_token_flashinfer,
)


# TODO: refactor to RouteStrategy base class
class EntropyStrategy(GenerationStrategy):
    """
    Entropy-based routing strategy adapted for HybridBackend.

    Routes to LLM when SLM's output entropy exceeds threshold.
    State management is delegated to the backend.
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
        """Generate using entropy-based routing with HybridBackend."""
        if not isinstance(hybrid_backend, HybridBackend):
            raise ValueError("EntropyStrategy requires a HybridBackend instance.")

        # 1. Initialize Session
        # Use a unique ID to let backends manage their own KV caches
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
        # SLM: Must return logits to calculate initial entropy
        slm_logits_all = hybrid_backend.forward(BackendId.SLM, req_id, input_ids)
        current_slm_logits = slm_logits_all[-1, :]  # Logits for the first new token

        # LLM: Just prefill to warm up the cache (return value ignored for now)
        _ = hybrid_backend.forward(BackendId.LLM, req_id, input_ids)

        # Track global generation state
        generated_ids = list(input_ids)

        # Track synchronization state: how many tokens have been fed to each model?
        # Initially, both have processed the prompt.
        slm_synced_len = prompt_len
        llm_synced_len = prompt_len

        eos_token_id = tokenizer.eos_token_id

        enable_stats_sync = kwargs.get("enable_stats_sync", False)
        if enable_stats_sync and device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        # Statistics
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0
        slm_llm_top1_set_equal = 0
        slm_llm_top2_set_equal = 0
        slm_llm_top3_set_equal = 0
        slm_llm_top1_exact = 0
        slm_llm_top2_exact = 0
        slm_llm_top3_exact = 0
        # Cumulative probability tracking for set_equal cases
        top2_set_equal_slm_cumprob = []
        top2_set_equal_llm_cumprob = []
        top3_set_equal_slm_cumprob = []
        top3_set_equal_llm_cumprob = []
        # Top-p sampling pool statistics
        topp_pool_set_equal = 0
        topp_pool_set_equal_slm_cumprob = []
        topp_pool_set_equal_llm_cumprob = []
        topp_pool_set_equal_sizes = []  # Size of the pool when equal
        all_slm_topp_pool_sizes = []  # All SLM pool sizes
        all_llm_topp_pool_sizes = []  # All LLM pool sizes

        # Helper to print/track
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

            # --- Decision Phase ---
            # Calculate entropy on current SLM logits
            entropy = calculate_token_entropy(current_slm_logits, temperature)
            # aleatoric_uncertainty, _ = compute_logu(current_slm_logits)

            use_llm = entropy >= threshold

            if not use_llm:
                # === SLM Generation Branch ===

                # 1. Sample
                next_token_tensor = sample_token_flashinfer(
                    current_slm_logits.unsqueeze(0), temperature, top_k, top_p
                )
                next_token_id = cast(int, next_token_tensor.item())
                slm_tokens += 1
                generated_ids.append(next_token_id)

                # 2. Update SLM State (Forward 1 step)
                # Input: [new_token] -> Output: logits for NEXT token
                current_slm_logits = hybrid_backend.forward(
                    BackendId.SLM, req_id, [next_token_id]
                )[-1, :]
                slm_synced_len += 1

                # LLM is now lagging behind by 1 token

                if _process_token(next_token_id, "slm", entropy, None):
                    break

            else:
                # === LLM Generation Branch ===

                # 0. Compare SLM/LLM top-k token sets (no sampling)
                # Get top-1, top-2, and top-3 tokens from SLM
                compare_k = min(3, current_slm_logits.numel())
                slm_top_ids = torch.topk(
                    current_slm_logits, k=compare_k
                ).indices.tolist()
                slm_top1_id = slm_top_ids[0] if len(slm_top_ids) >= 1 else None
                slm_top2_ids_set = set(slm_top_ids[:2]) if len(slm_top_ids) >= 2 else set()
                slm_top3_ids_set = set(slm_top_ids[:3])

                llm_next_logits = None

                # 1. LLM Catch-up
                # If SLM generated tokens while LLM was sleeping, feed them now.
                catchup_tokens = generated_ids[llm_synced_len:]

                if catchup_tokens:
                    # Feed missing history. The return value is the logits for the
                    # NEXT token (the one we are about to generate).
                    llm_next_logits = hybrid_backend.forward(
                        BackendId.LLM, req_id, catchup_tokens
                    )[-1, :]
                    llm_synced_len += len(catchup_tokens)
                else:
                    # Rare case: consecutive LLM calls or first step
                    # We need logits but didn't run forward in catch-up.
                    # This happens if llm_consecutive_tokens > 1 and we are in the loop below,
                    # but for the *first* token of the block, catchup_tokens might be empty
                    # only if we just switched? No, if we switched, slm must have generated something.
                    # This branch is defensive.
                    pass

                # 2. LLM Consecutive Generation
                hit_eos = False
                compare_first_token = True
                for _ in range(llm_consecutive_tokens):
                    if len(generated_ids) - prompt_len >= max_new_tokens:
                        break

                    # If we don't have logits (e.g. 2nd token in block), run forward
                    # If llm_next_logits is already set from catch-up, reuse it.
                    # So this loop only calls forward() for the 2nd, 3rd, ... tokens.
                    if llm_next_logits is None:
                        # Forward the PREVIOUS token to get logits for CURRENT
                        llm_next_logits = hybrid_backend.forward(
                            BackendId.LLM, req_id, [generated_ids[-1]]
                        )[-1, :]
                        llm_synced_len += 1

                    if compare_first_token:
                        compare_k = min(3, llm_next_logits.numel())
                        llm_top_ids = torch.topk(
                            llm_next_logits, k=compare_k
                        ).indices.tolist()

                        # Compute probability distributions (for cumulative probability calculation)
                        slm_probs = F.softmax(current_slm_logits / temperature, dim=-1)
                        llm_probs = F.softmax(llm_next_logits / temperature, dim=-1)

                        # Get top-p sampling pools
                        def get_topp_pool(probs, p=0.95):
                            """Get the set of token indices in the top-p sampling pool."""
                            sorted_probs, sorted_indices = torch.sort(
                                probs, descending=True
                            )
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            # Find tokens within top-p threshold
                            mask = cumsum_probs <= p
                            # Always include at least the top token
                            if not mask.any():
                                mask[0] = True
                            # Include one more token to reach p (the one that crosses the threshold)
                            else:
                                cutoff_idx = mask.sum().item()
                                if cutoff_idx < len(mask):
                                    mask[cutoff_idx] = True
                            selected_indices = sorted_indices[mask]
                            return set(selected_indices.tolist())

                        slm_topp_pool = get_topp_pool(slm_probs, top_p)
                        llm_topp_pool = get_topp_pool(llm_probs, top_p)

                        # Record pool sizes for all comparisons
                        all_slm_topp_pool_sizes.append(len(slm_topp_pool))
                        all_llm_topp_pool_sizes.append(len(llm_topp_pool))

                        # Compare top-1
                        llm_top1_id = llm_top_ids[0] if len(llm_top_ids) >= 1 else None
                        if slm_top1_id is not None and llm_top1_id == slm_top1_id:
                            slm_llm_top1_set_equal += 1
                            slm_llm_top1_exact += 1  # top-1: set_equal = exact

                        # Compare top-2 (set_equal: same set, exact: same order)
                        llm_top2_ids_set = set(llm_top_ids[:2]) if len(llm_top_ids) >= 2 else set()
                        # Set equal: compare sets (unordered)
                        if slm_top2_ids_set and slm_top2_ids_set == llm_top2_ids_set:
                            slm_llm_top2_set_equal += 1
                            # Calculate cumulative probability for top-2
                            slm_cumprob = sum(slm_probs[idx].item() for idx in slm_top_ids[:2])
                            llm_cumprob = sum(llm_probs[idx].item() for idx in llm_top_ids[:2])
                            top2_set_equal_slm_cumprob.append(slm_cumprob)
                            top2_set_equal_llm_cumprob.append(llm_cumprob)
                        # Exact: compare as ordered lists
                        if len(slm_top_ids) >= 2 and len(llm_top_ids) >= 2:
                            if slm_top_ids[:2] == llm_top_ids[:2]:
                                slm_llm_top2_exact += 1

                        # Compare top-3 (set_equal: same set, exact: same order)
                        llm_top3_ids_set = set(llm_top_ids[:3])
                        # Set equal: compare sets (unordered)
                        if slm_top3_ids_set == llm_top3_ids_set:
                            slm_llm_top3_set_equal += 1
                            # Calculate cumulative probability for top-3
                            slm_cumprob = sum(slm_probs[idx].item() for idx in slm_top_ids[:3])
                            llm_cumprob = sum(llm_probs[idx].item() for idx in llm_top_ids[:3])
                            top3_set_equal_slm_cumprob.append(slm_cumprob)
                            top3_set_equal_llm_cumprob.append(llm_cumprob)
                        # Exact: compare as ordered lists
                        if len(slm_top_ids) >= 3 and len(llm_top_ids) >= 3:
                            if slm_top_ids[:3] == llm_top_ids[:3]:
                                slm_llm_top3_exact += 1

                        # Compare top-p sampling pools
                        if slm_topp_pool == llm_topp_pool:
                            topp_pool_set_equal += 1
                            # Record pool size when equal
                            topp_pool_set_equal_sizes.append(len(slm_topp_pool))
                            # Calculate cumulative probability for the shared pool
                            slm_cumprob = sum(slm_probs[idx].item() for idx in slm_topp_pool)
                            llm_cumprob = sum(llm_probs[idx].item() for idx in llm_topp_pool)
                            topp_pool_set_equal_slm_cumprob.append(slm_cumprob)
                            topp_pool_set_equal_llm_cumprob.append(llm_cumprob)

                    # Sample
                    next_token_tensor = sample_token_flashinfer(
                        llm_next_logits.unsqueeze(0), temperature, top_k, top_p
                    )
                    next_token_id = next_token_tensor.item()
                    compare_first_token = False
                    llm_tokens += 1
                    generated_ids.append(next_token_id)

                    # Consume logits
                    llm_next_logits = None

                    if _process_token(next_token_id, "llm", entropy, None):
                        hit_eos = True
                        break

                # Note: llm_synced_len is NOT incremented for the very last generated token yet
                # because we haven't called forward on it (llm_next_logits is None).
                # This is fine, it will be handled in the next Catch-up or Sync phase.

                if hit_eos:
                    break

            # --- Synchronization Phase (SLM Catch-up) ---
            # SLM must always be up-to-date to calculate entropy for the NEXT step.
            # If LLM generated tokens, SLM is lagging.

            missing_tokens = generated_ids[slm_synced_len:]
            if missing_tokens:
                # Feed all new tokens to SLM
                # The last logit corresponds to the prediction for the upcoming token
                slm_logits_all = hybrid_backend.forward(
                    BackendId.SLM, req_id, missing_tokens
                )
                current_slm_logits = slm_logits_all[-1, :]
                slm_synced_len += len(missing_tokens)

        # 4. Cleanup
        hybrid_backend.free(BackendId.SLM, req_id)
        hybrid_backend.free(BackendId.LLM, req_id)

        if enable_stats_sync and device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Calculate average cumulative probabilities for set_equal cases
        def calc_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        # Statistics
        stats = {
            "total_tokens": len(generated_ids) - prompt_len,
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "slm_llm_top1_set_equal": slm_llm_top1_set_equal,
            "slm_llm_top2_set_equal": slm_llm_top2_set_equal,
            "slm_llm_top3_set_equal": slm_llm_top3_set_equal,
            "slm_llm_top1_exact": slm_llm_top1_exact,
            "slm_llm_top2_exact": slm_llm_top2_exact,
            "slm_llm_top3_exact": slm_llm_top3_exact,
            "top2_set_equal_avg_cumprob": {
                "slm": round(calc_avg(top2_set_equal_slm_cumprob), 4),
                "llm": round(calc_avg(top2_set_equal_llm_cumprob), 4),
                "min": round(
                    calc_avg(
                        [
                            min(s, l)
                            for s, l in zip(
                                top2_set_equal_slm_cumprob, top2_set_equal_llm_cumprob
                            )
                        ]
                    ),
                    4,
                )
                if top2_set_equal_slm_cumprob
                else 0.0,
            },
            "top3_set_equal_avg_cumprob": {
                "slm": round(calc_avg(top3_set_equal_slm_cumprob), 4),
                "llm": round(calc_avg(top3_set_equal_llm_cumprob), 4),
                "min": round(
                    calc_avg(
                        [
                            min(s, l)
                            for s, l in zip(
                                top3_set_equal_slm_cumprob, top3_set_equal_llm_cumprob
                            )
                        ]
                    ),
                    4,
                )
                if top3_set_equal_slm_cumprob
                else 0.0,
            },
            "topp_pool_set_equal": topp_pool_set_equal,
            "topp_pool_set_equal_avg_cumprob": {
                "slm": round(calc_avg(topp_pool_set_equal_slm_cumprob), 4),
                "llm": round(calc_avg(topp_pool_set_equal_llm_cumprob), 4),
                "min": round(
                    calc_avg(
                        [
                            min(s, l)
                            for s, l in zip(
                                topp_pool_set_equal_slm_cumprob,
                                topp_pool_set_equal_llm_cumprob,
                            )
                        ]
                    ),
                    4,
                )
                if topp_pool_set_equal_slm_cumprob
                else 0.0,
            },
            "topp_pool_sizes": {
                "slm_avg": round(calc_avg(all_slm_topp_pool_sizes), 2),
                "llm_avg": round(calc_avg(all_llm_topp_pool_sizes), 2),
                "when_equal_avg": round(calc_avg(topp_pool_set_equal_sizes), 2),
            },
            "elapsed_time": end_time - start_time,
            "threshold": threshold,
            "llm_consecutive_tokens": llm_consecutive_tokens,
        }

        return tokenizer.decode(
            generated_ids[prompt_len:], skip_special_tokens=True
        ), stats
