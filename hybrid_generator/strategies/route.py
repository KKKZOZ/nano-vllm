import random
import time
import uuid
from typing import Tuple, cast

import torch

from hybrid_generator.backends import ModelBackend
from hybrid_generator.strategies.base import GenerationStrategy
from hybrid_generator.strategies.metrics import LiveMetricsTracker
from hybrid_generator.strategies.utils import (
    calculate_token_entropy,
    compute_logu,
    sample_token,
)


# TODO: refactor to RouteStrategy base class
class EntropyStrategy(GenerationStrategy):
    """
    Entropy-based routing strategy adapted for ModelBackend.

    Routes to LLM when SLM's output entropy exceeds threshold.
    State management is delegated to the backend.
    """

    def generate(
        self,
        slm: ModelBackend,
        llm: ModelBackend,
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
        """Generate using entropy-based routing with ModelBackend."""

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
        slm_logits_all = slm.forward(req_id, input_ids)
        current_slm_logits = slm_logits_all[-1, :]  # Logits for the first new token

        # LLM: Just prefill to warm up the cache (return value ignored for now)
        _ = llm.forward(req_id, input_ids)

        # Track global generation state
        generated_ids = list(input_ids)

        # Track synchronization state: how many tokens have been fed to each model?
        # Initially, both have processed the prompt.
        slm_synced_len = prompt_len
        llm_synced_len = prompt_len

        eos_token_id = tokenizer.eos_token_id

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        # Statistics
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

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
                next_token_tensor, _ = sample_token(
                    current_slm_logits.unsqueeze(0), temperature, top_k, top_p, min_p
                )
                next_token_id = cast(int, next_token_tensor.item())
                slm_tokens += 1
                generated_ids.append(next_token_id)

                # 2. Update SLM State (Forward 1 step)
                # Input: [new_token] -> Output: logits for NEXT token
                current_slm_logits = slm.forward(req_id, [next_token_id])[-1, :]
                slm_synced_len += 1

                # LLM is now lagging behind by 1 token

                if _process_token(next_token_id, "slm", entropy, None):
                    break

            else:
                # === LLM Generation Branch ===

                # 1. LLM Catch-up (补课)
                # If SLM generated tokens while LLM was sleeping, feed them now.
                catchup_tokens = generated_ids[llm_synced_len:]

                if catchup_tokens:
                    # Feed missing history. The return value is the logits for the
                    # NEXT token (the one we are about to generate).
                    llm_next_logits = llm.forward(req_id, catchup_tokens)[-1, :]
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
                for _ in range(llm_consecutive_tokens):
                    if len(generated_ids) - prompt_len >= max_new_tokens:
                        break

                    # If we don't have logits (e.g. 2nd token in block), run forward
                    if llm_next_logits is None:
                        # Forward the PREVIOUS token to get logits for CURRENT
                        llm_next_logits = llm.forward(req_id, [generated_ids[-1]])[
                            -1, :
                        ]
                        llm_synced_len += 1

                    # Sample
                    next_token_tensor, _ = sample_token(
                        llm_next_logits.unsqueeze(0), temperature, top_k, top_p, min_p
                    )
                    next_token_id = next_token_tensor.item()
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
                slm_logits_all = slm.forward(req_id, missing_tokens)
                current_slm_logits = slm_logits_all[-1, :]
                slm_synced_len += len(missing_tokens)

        # 4. Cleanup
        slm.free(req_id)
        llm.free(req_id)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Statistics
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
