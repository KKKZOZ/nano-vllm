import random
import time
from typing import Tuple

import torch

from hybrid_generator.backends import BackendId, HybridBackend
from hybrid_generator.strategies.base import GenerationStrategy
from hybrid_generator.strategies.utils import sample_token_flashinfer


class SoloStrategy(GenerationStrategy):
    """
    Single-backend generation using HybridBackend.

    Generates tokens autoregressively with the specified backend (SLM or LLM).
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
        verbose: bool = False,
        enable_stats_sync: bool = False,
        backend_id: BackendId = BackendId.LLM,
        **kwargs,
    ) -> Tuple[str, dict]:
        """Generate text using only one backend."""
        if not isinstance(hybrid_backend, HybridBackend):
            raise ValueError("SoloStrategy requires a HybridBackend instance.")

        if not isinstance(backend_id, BackendId):
            raise ValueError("backend_id must be a BackendId enum value.")

        req_id = random.randint(0, 2**31 - 1)

        # Prefill to warm cache and get initial logits
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0].tolist()
        prompt_len = len(input_ids)

        logits = hybrid_backend.forward(backend_id, req_id, input_ids)
        current_logits = logits[-1, :]

        generated_ids = list(input_ids)
        eos_token_id = tokenizer.eos_token_id

        if enable_stats_sync and device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        while len(generated_ids) - prompt_len < max_new_tokens:
            next_token_tensor = sample_token_flashinfer(
                current_logits.unsqueeze(0), temperature, top_k, top_p
            )
            next_token_id = int(next_token_tensor.item())
            generated_ids.append(next_token_id)

            if verbose:
                print(
                    tokenizer.decode([next_token_id], skip_special_tokens=True),
                    end="",
                    flush=True,
                )

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            # Forward the newly generated token to obtain logits for the next step
            current_logits = hybrid_backend.forward(
                backend_id, req_id, [next_token_id]
            )[-1, :]

        hybrid_backend.free(backend_id, req_id)

        if enable_stats_sync and device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        stats = {
            "total_tokens": len(generated_ids) - prompt_len,
            "backend": backend_id.value,
            "elapsed_time": end_time - start_time,
            "decode_steps": len(generated_ids) - prompt_len,
        }

        return tokenizer.decode(
            generated_ids[prompt_len:], skip_special_tokens=True
        ), stats
