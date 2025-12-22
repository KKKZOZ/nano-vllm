from typing import cast
import time
import uuid

import torch

from hybrid_generator.backends import ModelBackend
from hybrid_generator.strategies.base import GenerationStrategy
from hybrid_generator.strategies.utils import sample_token


class SpeculativeStrategy(GenerationStrategy):
    """
    Speculative decoding strategy adapted for ModelBackend interface.

    This implementation delegates KV cache management to the backend,
    making it compatible with Transformers, vLLM, or SGLang backends.
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
        num_drafts: int = 4,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[str, dict]:
        """Generate using speculative decoding with ModelBackend."""

        # 1. Initialize Session
        # Use a unique ID to manage state in the backend
        req_id = str(uuid.uuid4())

        # 2. Prefill Phase
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0].tolist()
        prompt_len = len(input_ids)

        # Forward prompt to both models to populate initial KV cache
        # Backend returns logits for all input tokens
        _ = slm.forward(req_id, input_ids)
        llm_logits_all = llm.forward(req_id, input_ids)

        # Sample first token from LLM (using the last logit)
        token_id, _ = sample_token(
            llm_logits_all[-1, :].unsqueeze(0), temperature, top_k, top_p, min_p
        )

        # Update trackers
        generated_ids = [token_id.item()]
        curr_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(curr_text, end="", flush=True)

        eos_token_id = tokenizer.eos_token_id
        start_time = time.time()

        # Statistics
        draft_generated = 0
        draft_accepted = 0
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

        # Current "anchor" token (confirmed) to start drafting from
        current_token_id = cast(int, token_id.item())

        # Track the committed cache length (starts after prompt)
        # Note: Prefill output includes prompt_len positions.
        # But we haven't fed the first generated token 'current_token_id' into the cache yet.
        # So committed length is just prompt_len.
        cache_len = prompt_len

        # 3. Main Loop
        while len(generated_ids) < max_new_tokens:
            decode_steps += 1

            # --- A. Draft Generation (SLM) ---
            draft_tokens = []
            draft_probs = []

            # SLM needs to process the anchor token first?
            # In prefill, SLM processed prompt.
            # We need to feed 'current_token_id' to SLM to get prediction for draft_1.
            next_input_id = current_token_id

            for _ in range(num_drafts):
                # Forward 1 token -> Append to SLM cache -> Get 1 logit
                # Note: This automatically handles 'past_key_values' inside backend
                logits = slm.forward(req_id, [next_input_id])

                # Sample draft token
                next_token_tensor, probs = sample_token(
                    logits[-1, :].unsqueeze(0), temperature, top_k, top_p, min_p
                )
                next_input_id = next_token_tensor.item()

                draft_tokens.append(next_input_id)
                draft_probs.append(probs)

            draft_generated += len(draft_tokens)

            # --- B. Verification (LLM) ---
            # Input to LLM: [anchor_token, draft_1, draft_2, ..., draft_n]
            # LLM cache currently ends at 'prompt' (or previous verified point).
            # This forward call appends all these tokens to LLM cache.
            verify_input_ids = [current_token_id] + draft_tokens

            # Returns logits for [anchor, d1, d2, ...]
            # anchor's logit -> predicts d1
            # d1's logit -> predicts d2
            llm_logits_all = llm.forward(req_id, verify_input_ids)

            # --- C. Rejection Sampling ---
            accept_count = 0
            accepted_tokens_this_round = []
            bonus_token_id = None

            for i in range(num_drafts):
                draft_id = draft_tokens[i]

                # LLM prediction for this position comes from the PREVIOUS token
                # i=0: verify d1, need logit from verify_input_ids[0] (anchor)
                # i=1: verify d2, need logit from verify_input_ids[1] (d1)
                llm_prob_dist = torch.softmax(
                    llm_logits_all[i, :] / temperature, dim=-1
                )

                p_target = llm_prob_dist[draft_id].item()
                q_draft = draft_probs[i][0, draft_id].item()

                # Speculative Sampling Formula
                acceptance_prob = min(1.0, p_target / (q_draft + 1e-10))

                if torch.rand(1).item() < acceptance_prob:
                    accepted_tokens_this_round.append(draft_id)
                    accept_count += 1
                else:
                    # Rejected: Resample from adjusted distribution
                    # p'(x) = norm(max(0, p(x) - q(x)))
                    # Note: We need the full probability distribution from SLM for this step,
                    # but draft_probs stored above usually is just the selected prob or full dist?
                    # Assuming draft_probs stores full distribution or we re-approximate.
                    # For strict correctness, SLM sample_token should return full probs.
                    # Here we use a simplified rejection for brevity or assume draft_probs is full.

                    slm_prob_dist = draft_probs[i][0]  # Assuming shape [1, vocab]
                    adjusted_probs = torch.clamp(llm_prob_dist - slm_prob_dist, min=0.0)
                    if adjusted_probs.sum() > 0:
                        adjusted_probs /= adjusted_probs.sum()
                        bonus_token_id = torch.multinomial(adjusted_probs, 1).item()
                    else:
                        # Fallback if numerical issues
                        bonus_token_id = torch.multinomial(llm_prob_dist, 1).item()

                    llm_tokens += 1
                    break

            draft_accepted += accept_count
            slm_tokens += accept_count

            # --- D. Bonus Token ---
            # If all drafts accepted, sample one more from LLM's last output
            if accept_count == num_drafts:
                # Last logit corresponds to prediction after the last draft
                last_logit = llm_logits_all[-1, :]
                bonus_token_tensor, _ = sample_token(
                    last_logit.unsqueeze(0), temperature, top_k, top_p, min_p
                )
                bonus_token_id = bonus_token_tensor.item()
                llm_tokens += 1

            # --- E. Rollback & State Update ---

            # We confirmed: [anchor_token] + [accepted_tokens]
            # The cache currently contains:
            #   SLM: prompt + [anchor] + [drafts] (len = cache_len + 1 + num_drafts)
            #   LLM: prompt + [anchor] + [drafts] (len = cache_len + 1 + num_drafts)

            # The valid cache should end after the last ACCEPTED token.
            # New confirmed cache length = cache_len + 1 (anchor) + accept_count
            new_cache_len = cache_len + 1 + accept_count

            # Rollback both models to discard rejected drafts
            # Even if we accepted all, we might need to sync if SLM ran ahead differently (rare here)
            # or simply to confirm the state logic.
            # If we rejected, this cuts off the bad branch.
            slm.rollback(req_id, new_cache_len)
            llm.rollback(req_id, new_cache_len)

            cache_len = new_cache_len

            # Update generated IDs list
            # Note: anchor was already 'generated' in previous step (or prefill),
            # so we only append newly accepted drafts + bonus
            to_append = accepted_tokens_this_round
            if bonus_token_id is not None:
                to_append.append(bonus_token_id)
                # Note: The bonus token is NOT in the cache yet.
                # It will be the 'anchor' for the next loop.
                current_token_id = bonus_token_id
            else:
                # Should not happen in standard speculative logic (always produce 1 extra),
                # but handle just in case
                current_token_id = accepted_tokens_this_round[-1]

            generated_ids.extend(to_append)

            # Print
            new_text = tokenizer.decode(to_append, skip_special_tokens=True)
            print(new_text, end="", flush=True)

            if eos_token_id in to_append:
                break

        # 4. Cleanup
        slm.free(req_id)
        llm.free(req_id)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        stats = {
            "total_tokens": len(generated_ids),
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "elapsed_time": end_time - start_time,
            "draft_generated": draft_generated,
            "draft_accepted": draft_accepted,
        }

        return tokenizer.decode(generated_ids, skip_special_tokens=True), stats
