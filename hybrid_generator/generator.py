"""
HybridGenerator: Main class for hybrid SLM/LLM generation.

This module provides a unified interface for generating text using
different strategies that combine a small language model (SLM) with
a large language model (LLM).
"""

import time
from typing import Literal

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from hybrid_generator.backends import HFBackend, NanovLLMBackend
from hybrid_generator.profiling import ProfileResult
from hybrid_generator.strategies import (
    EntropyStrategy,
    SpeculativeStrategy,
    calculate_token_entropy,
    compute_logu,
    sample_token,
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
        slm_model_id: str,
        llm_model_id: str,
        device: str = "cuda",
        dtype=torch.float16,
        verbose: bool = False,
        report_live_metrics: bool = False,
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

        # Load tokenizer (use LLM's tokenizer)
        print(f"Loading tokenizer from {llm_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

        # Load backend
        print(f"Loading SLM: {slm_model_id}")
        # self.slm = AutoModelForCausalLM.from_pretrained(
        #     slm_model_id, torch_dtype=dtype
        # ).to(device)
        # self.slm.eval()
        self.slm = NanovLLMBackend(slm_model_id, device=device, dtype=dtype)

        print(f"Loading LLM: {llm_model_id}")
        # self.llm = AutoModelForCausalLM.from_pretrained(
        #     llm_model_id, torch_dtype=dtype
        # ).to(device)
        # self.llm.eval()
        self.llm = NanovLLMBackend(llm_model_id, device=device, dtype=dtype)

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
        # Routing parameters
        threshold: float = 0.5,
        routing_metric: Literal["uncertainty", "entropy"] = "uncertainty",
        enable_routing: bool = False,
    ) -> ProfileResult:
        """
        Generate text with detailed profiling of each token.

        This method generates text while recording detailed statistics for each token,
        including uncertainty, entropy, and which model was used. The result includes
        comprehensive analysis of the distribution of these metrics.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            threshold: Threshold for routing (only used if enable_routing=True)
            routing_metric: Metric to use for routing ("uncertainty" or "entropy")
            enable_routing: Whether to enable routing between SLM and LLM
                - False (default): Use only SLM for all tokens, profile SLM's complete
                  uncertainty/entropy distribution. Best for finding optimal threshold.
                - True: Use routing strategy, profile actual runtime behavior with
                  the given threshold.

        Returns:
            ProfileResult object containing generated text and detailed statistics

        Examples:
            >>> # Profile SLM to find optimal threshold
            >>> profile = generator.generate_with_profile(
            ...     prompt="Explain quantum computing",
            ...     max_new_tokens=500,
            ...     enable_routing=False  # Only use SLM
            ... )
            >>> profile.print_summary()
            >>> # If top 30% has uncertainty > 0.5, set threshold=0.5

            >>> # Profile actual runtime with chosen threshold
            >>> profile = generator.generate_with_profile(
            ...     prompt="Explain quantum computing",
            ...     max_new_tokens=100,
            ...     threshold=0.5,
            ...     routing_metric="uncertainty",
            ...     enable_routing=True  # Use routing
            ... )
        """
        # Initialize profiling result
        mode = "slm_only" if not enable_routing else f"routed_{routing_metric}"
        profile = ProfileResult(
            generated_text="", strategy=f"profiled_{mode}", total_time=0.0
        )

        # Initialize KV caches
        slm_cache = DynamicCache(config=self.slm.config)
        llm_cache = DynamicCache(config=self.llm.config) if enable_routing else None

        # Prefill
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        cache_position = torch.arange(prompt_len, device=self.device, dtype=torch.long)

        # Prefill SLM (always needed)
        _ = self.slm(
            **inputs,
            past_key_values=slm_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Prefill LLM only if routing is enabled
        if enable_routing:
            _ = self.llm(
                **inputs,
                past_key_values=llm_cache,
                use_cache=True,
                cache_position=cache_position,
            )

        generated_ids = inputs["input_ids"]
        offset = prompt_len

        eos_token_ids = (
            [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        # Statistics
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

        if enable_routing:
            print(
                f"Generating with profiling (routing by {routing_metric}, threshold={threshold})..."
            )
        else:
            print("Generating with profiling (SLM only - no routing)...")

        # Main generation loop
        while generated_ids.shape[1] - prompt_len < max_new_tokens:
            decode_steps += 1

            # Get SLM prediction
            cache_pos = torch.tensor([offset], device=self.device, dtype=torch.long)
            slm_outputs = self.slm(
                input_ids=generated_ids[:, -1:],
                past_key_values=slm_cache,
                use_cache=True,
                cache_position=cache_pos,
            )

            # Calculate metrics from SLM
            slm_logits = slm_outputs.logits[:, -1, :]
            aleatoric_uncertainty, epistemic_uncertainty = compute_logu(slm_logits)
            entropy = calculate_token_entropy(slm_logits, temperature)
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()

            # Decide which model to use
            if enable_routing:
                # Use routing strategy
                if routing_metric == "uncertainty":
                    use_llm = aleatoric_uncertainty >= threshold
                else:  # entropy
                    use_llm = entropy >= threshold
            else:
                # No routing - always use SLM
                use_llm = False

            # Generate token and get probabilities
            if use_llm:
                # Use LLM
                llm_outputs = self.llm(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=llm_cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                next_token, probs = sample_token(
                    llm_outputs.logits[:, -1, :], temperature, top_k, top_p, min_p
                )
                model_used = "llm"
                llm_tokens += 1
            else:
                # Use SLM
                next_token, probs = sample_token(
                    slm_logits, temperature, top_k, top_p, min_p
                )
                # Update LLM cache only if routing is enabled
                if enable_routing:
                    _ = self.llm(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=llm_cache,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                model_used = "slm"
                slm_tokens += 1

            # Get top-k probabilities for this token
            top_k_probs_tensor, top_k_indices = torch.topk(
                probs[0], min(5, probs.shape[-1])
            )
            top_k_probs = [
                (int(top_k_indices[i]), float(top_k_probs_tensor[i]))
                for i in range(len(top_k_indices))
            ]

            # Record token profile
            token_id = next_token.item()
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)

            profile.add_token(
                token_id=token_id,
                token_text=token_text,
                position=offset - prompt_len,
                aleatoric_uncertainty=aleatoric_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                entropy=entropy,
                model_used=model_used,
                top_k_probs=top_k_probs,
            )

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            offset += 1

            # Print token (with indicator for which model was used) if verbose mode is enabled
            if self.verbose:
                indicator = "🔵" if model_used == "llm" else "🟢"
                print(f"{token_text}", end="", flush=True)

            if token_id in eos_token_ids:
                break

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Finalize profile
        profile.generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        profile.total_time = end_time - start_time
        profile.stats = {
            "total_tokens": len(profile.tokens),
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "routing_metric": routing_metric,
            "threshold": threshold,
            "enable_routing": enable_routing,
        }

        print(
            f"\n\nGenerated {len(profile.tokens)} tokens in {profile.total_time:.2f}s"
        )
        print(f"Speed: {len(profile.tokens) / max(profile.total_time, 1e-9):.2f} tok/s")

        return profile
