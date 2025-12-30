"""
HybridGenerator: Main class for hybrid SLM/LLM generation.

This module provides a unified interface for generating text using
different strategies that combine a small language model (SLM) with
a large language model (LLM).
"""

import random
import time
from typing import Literal, cast

import torch
from transformers import AutoTokenizer

from hybrid_generator.backends import NanovLLMBackend
from hybrid_generator.profiling import ProfileResult
from hybrid_generator.strategies import (
    EntropyStrategy,
    SpeculativeStrategy,
    calculate_token_entropy,
    compute_logu,
    sample_token,
)
from hybrid_generator.strategies.utils import (
    sample_token_flashinfer,
)


class HybridGenerator:
    """
    Hybrid generator that combines SLM and LLM with multiple strategies.

    This class provides a unified interface for text generation using different
    strategies that intelligently combine a fast small model (SLM) with a more
    accurate large model (LLM).

    Supported strategies:
    - speculative: Standard speculative decoding with sampling
    - uncertainty: Route to LLM when aleatoric uncertainty is high
    - entropy: Route to LLM when entropy is high

    Example:
        >>> generator = HybridGenerator(
        ...     slm_model_id="Qwen/Qwen3-1.7B",
        ...     llm_model_id="Qwen/Qwen3-8B"
        ... )
        >>> result = generator.generate(
        ...     prompt="Write a story about AI",
        ...     strategy="speculative",
        ...     max_new_tokens=200
        ... )
    """

    def __init__(
        self,
        slm_model_id: str | None,
        llm_model_id: str | None,
        slm_memory_usage: float = 0.4,
        llm_memory_usage: float = 0.4,
        device: str = "cuda",
        dtype=torch.float16,
        verbose: bool = False,
        report_live_metrics: bool = False,
        enable_stats_sync: bool = False,
    ):
        """
        Initialize the hybrid generator.

        Args:
            slm_model_id: HuggingFace model ID for the small/fast model
            llm_model_id: HuggingFace model ID for the large/accurate model
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type for models (e.g., torch.float16, torch.bfloat16)
            verbose: Whether to print generated tokens in real-time (default: False)
            report_live_metrics: Whether to report live metrics every 100 tokens (default: False)
        """
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.report_live_metrics = report_live_metrics
        self.enable_stats_sync = enable_stats_sync

        if slm_model_id is None and llm_model_id is None:
            raise ValueError(
                "At least one of slm_model_id or llm_model_id must be provided."
            )

        # Load tokenizer
        if llm_model_id is not None:
            print(f"Loading tokenizer from {llm_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        else:
            print(f"Loading tokenizer from {slm_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(slm_model_id)

        # Load backend
        if slm_model_id is not None:
            print(f"Loading SLM: {slm_model_id}")
            self.slm = NanovLLMBackend(
                slm_model_id,
                device=device,
                dtype=dtype,
                gpu_memory_utilization=slm_memory_usage,
                max_num_seqs=1,
                enable_stats_sync=enable_stats_sync,
            )

        if llm_model_id is not None:
            print(f"Loading LLM: {llm_model_id}")
            # self.llm = AutoModelForCausalLM.from_pretrained(
            #     llm_model_id, torch_dtype=dtype
            # ).to(device)
            # self.llm.eval()
            self.llm = NanovLLMBackend(
                llm_model_id,
                device=device,
                dtype=dtype,
                max_num_seqs=1,
                gpu_memory_utilization=llm_memory_usage,
                enable_stats_sync=enable_stats_sync,
            )

        # Initialize strategies
        self.strategies = {
            "speculative": SpeculativeStrategy(),
            # "uncertainty": UncertaintyStrategy(),
            "entropy": EntropyStrategy(),
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        strategy: Literal["speculative", "uncertainty", "entropy"] = "speculative",
        max_new_tokens: int = 2000,
        # Sampling parameters
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        min_p: float = 0.0,
        # Strategy-specific parameters
        num_drafts: int = 4,  # for speculative
        threshold: float = 0.5,  # for uncertainty and entropy
    ) -> tuple[str, dict]:
        """
        Generate text using the specified strategy.

        Args:
            prompt: Input prompt text
            strategy: Generation strategy to use:
                - "speculative": SLM drafts tokens, LLM verifies in parallel
                - "uncertainty": Route to LLM when SLM is uncertain
                - "entropy": Route to LLM when SLM output has high entropy
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            min_p: Minimum probability threshold (0.0 = disabled)
            num_drafts: Number of draft tokens for speculative strategy
            threshold: Threshold for uncertainty/entropy routing strategies

        Returns:
            Tuple of (generated_text, statistics_dict) where:
            - generated_text: Generated text (including the prompt)
            - statistics_dict: Dictionary containing generation statistics

        Raises:
            ValueError: If an unknown strategy is specified
        """
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available strategies: {list(self.strategies.keys())}"
            )

        # Get the strategy implementation
        strategy_impl = self.strategies[strategy]

        # Generate using the strategy
        result, stats = strategy_impl.generate(
            slm=self.slm,
            llm=self.llm,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            device=self.device,
            num_drafts=num_drafts,
            threshold=threshold,
            verbose=self.verbose,
            report_live_metrics=self.report_live_metrics,
            enable_stats_sync=self.enable_stats_sync,
        )

        # Print statistics
        self._print_stats(stats, strategy)

        return result, stats

    def _print_stats(self, stats: dict, strategy: str):
        """Print generation statistics in a formatted way."""
        elapsed = stats["elapsed_time"]
        speed = stats["total_tokens"] / max(elapsed, 1e-9)

        print(f"\n\n{'=' * 60}")
        print(f"Strategy: {strategy}")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"Total tokens generated: {stats['total_tokens']}")
        print(f"Speed: {speed:.2f} tok/s")
        print(f"Decode steps: {stats['decode_steps']}")

        if strategy == "speculative":
            draft_gen = stats.get("draft_generated", 0)
            draft_acc = stats.get("draft_accepted", 0)
            if draft_gen > 0:
                acceptance_rate = draft_acc / draft_gen
                print(
                    f"Draft tokens: {draft_gen}, "
                    f"Accepted: {draft_acc}, "
                    f"Acceptance rate: {acceptance_rate:.2%}"
                )
            print(
                f"SLM tokens: {stats['slm_tokens']}, LLM tokens: {stats['llm_tokens']}"
            )
        elif strategy in ["uncertainty", "entropy"]:
            print(
                f"SLM tokens: {stats['slm_tokens']}, LLM tokens: {stats['llm_tokens']}"
            )
            if stats["total_tokens"] > 0:
                slm_ratio = stats["slm_tokens"] / stats["total_tokens"]
                print(f"SLM usage: {slm_ratio:.2%}")
            if "threshold" in stats:
                print(f"Threshold: {stats['threshold']}")

        print(f"{'=' * 60}\n")

    @torch.inference_mode()
    def generate_with_profile(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        # Sampling parameters
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        min_p: float = 0.0,
    ) -> ProfileResult:
        """
        Generate text with detailed profiling of each token.

        This method generates text using only the SLM while recording detailed statistics
        for each token, including uncertainty and entropy. The result includes
        comprehensive analysis of the distribution of these metrics.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold

        Returns:
            ProfileResult object containing generated text and detailed statistics

        """
        # Initialize profiling result
        profile = ProfileResult(
            generated_text="", strategy="profiled_slm_only", total_time=0.0
        )

        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"][0].tolist()
        prompt_len = len(prompt_ids)
        generated_ids = list(prompt_ids)

        eos_token_ids = (
            [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
        )

        # Initialize backend session
        req_id = random.randint(0, 2**31 - 1)
        slm_logits = self.slm.forward(req_id, prompt_ids)

        if self.enable_stats_sync and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        # Statistics
        slm_tokens = 0
        decode_steps = 0

        print("Generating with profiling (SLM only)...")

        # Main generation loop
        next_logits = slm_logits
        while len(generated_ids) - prompt_len < max_new_tokens:
            decode_steps += 1

            # Use logits from previous step to sample next token
            token_logits = next_logits[-1, :]
            aleatoric_uncertainty, epistemic_uncertainty = compute_logu(token_logits)
            entropy = calculate_token_entropy(token_logits, temperature)
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()

            top_k_probs = None
            next_token = sample_token_flashinfer(
                token_logits.unsqueeze(0), temperature, top_k, top_p
            )
            slm_tokens += 1

            # Get top-k probabilities for this token
            # top_k_probs_tensor, top_k_indices = torch.topk(
            #     probs[0], min(5, probs.shape[-1])
            # )
            # top_k_probs = [
            #     (int(top_k_indices[i]), float(top_k_probs_tensor[i]))
            #     for i in range(len(top_k_indices))
            # ]

            # Record token profile
            token_id = cast(int, next_token.item())
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)

            profile.add_token(
                token_id=token_id,
                token_text=token_text,
                position=len(generated_ids) - prompt_len,
                aleatoric_uncertainty=aleatoric_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                entropy=entropy,
                model_used="slm",
                top_k_probs=top_k_probs,
            )

            generated_ids.append(token_id)

            # Print token (with indicator for which model was used) if verbose mode is enabled
            if self.verbose:
                # indicator = "🔵" if model_used == "llm" else "🟢"
                print(f"{token_text}", end="", flush=True)

            if token_id in eos_token_ids:
                break

            # Prepare logits for next step
            next_logits = self.slm.forward(req_id, [token_id])

        if self.enable_stats_sync and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Finalize profile
        profile.generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        profile.total_time = end_time - start_time
        profile.stats = {
            "total_tokens": len(profile.tokens),
            "slm_tokens": slm_tokens,
            "decode_steps": decode_steps,
            "enable_routing": False,
        }

        print(
            f"\n\nGenerated {len(profile.tokens)} tokens in {profile.total_time:.2f}s"
        )
        print(f"Speed: {len(profile.tokens) / max(profile.total_time, 1e-9):.2f} tok/s")

        return profile

    @torch.inference_mode()
    def simple_generate_with_slm(
        self,
        prompt: str,
        max_new_tokens: int = 2000,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        min_p: float = 0.0,
    ) -> tuple[str, dict]:
        """
        Generate text using only the SLM backend with simple sampling.

        This helper avoids any routing logic and always decodes with the SLM.
        It returns the generated text alongside basic stats including decode speed.
        """
        if not getattr(self, "slm", None):
            raise ValueError("SLM backend is not initialized.")

        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"][0].tolist()
        prompt_len = len(prompt_ids)
        generated_ids = list(prompt_ids)

        eos_token_ids = (
            [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
        )

        # Initialize backend session and prefill
        req_id = random.randint(0, 2**31 - 1)
        logits = self.slm.forward(req_id, prompt_ids)

        if self.enable_stats_sync and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        slm_tokens = 0
        decode_steps = 0

        while len(generated_ids) - prompt_len < max_new_tokens:
            decode_steps += 1

            # Use last logits to sample next token
            token_logits = logits[-1, :]
            next_token, _ = sample_token(
                token_logits.unsqueeze(0), temperature, top_k, top_p, min_p
            )
            token_id = cast(int, next_token.item())
            generated_ids.append(token_id)
            slm_tokens += 1

            if self.verbose:
                print(
                    self.tokenizer.decode([token_id], skip_special_tokens=True),
                    end="",
                    flush=True,
                )

            if token_id in eos_token_ids:
                break

            # Decode next step with SLM
            logits = self.slm.forward(req_id, [token_id])

        if self.enable_stats_sync and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        total_new_tokens = len(generated_ids) - prompt_len
        stats = {
            "elapsed_time": end_time - start_time,
            "total_tokens": total_new_tokens,
            "slm_tokens": slm_tokens,
            "decode_steps": decode_steps,
        }

        self._print_stats(stats, strategy="slm_only")

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, stats

    def report_backend_stats(self) -> dict:
        """Report statistics from both SLM and LLM backends."""
        return {
            "slm_stats": self.slm.report_stats(),
            "llm_stats": self.llm.report_stats(),
        }
