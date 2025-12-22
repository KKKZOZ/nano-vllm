"""
Generation strategies for hybrid SLM/LLM decoding.

This module contains implementations of three strategies:
1. SpeculativeStrategy: Standard speculative decoding with sampling
2. UncertaintyStrategy: Route to LLM when aleatoric uncertainty is high
3. EntropyStrategy: Route to LLM when entropy is high
"""

import time
from abc import ABC, abstractmethod

import torch
from transformers.cache_utils import DynamicCache

# Allow imports when executed as a package or as standalone scripts
try:
    from .utils import calculate_token_entropy, compute_logu, sample_token
except ImportError:
    from utils import calculate_token_entropy, compute_logu, sample_token


class LiveMetricsTracker:
    """Tracks and reports live generation metrics every N tokens."""

    def __init__(self, report_interval: int = 100):
        self.report_interval = report_interval
        self.slm_tokens = 0
        self.llm_tokens = 0
        self.total_tokens = 0
        self.entropy_values = []
        self.uncertainty_values = []
        # Store recent tokens for display
        self.recent_tokens = []  # List of (token_text, model_used, entropy, uncertainty)
        self.max_recent_tokens = report_interval  # Keep last N tokens

    def add_token(
        self,
        model_used: str,
        token_text: str = None,
        entropy: float = None,
        uncertainty: float = None,
    ):
        """Record a generated token."""
        if model_used == "slm":
            self.slm_tokens += 1
        elif model_used == "llm":
            self.llm_tokens += 1

        self.total_tokens += 1

        if entropy is not None:
            self.entropy_values.append(entropy)
        if uncertainty is not None:
            self.uncertainty_values.append(uncertainty)

        # Store token information
        if token_text is not None:
            self.recent_tokens.append(
                {
                    "text": token_text,
                    "model": model_used,
                    "entropy": entropy,
                    "uncertainty": uncertainty,
                }
            )
            # Keep only the last N tokens
            if len(self.recent_tokens) > self.max_recent_tokens:
                self.recent_tokens.pop(0)

    def should_report(self) -> bool:
        """Check if we should print a progress report."""
        return self.total_tokens > 0 and self.total_tokens % self.report_interval == 0

    def print_progress(self):
        """Print current generation statistics."""
        print(f"\n{'=' * 70}")
        print(f"Progress Report - {self.total_tokens} tokens generated")
        print(f"{'=' * 70}")
        print(
            f"SLM tokens: {self.slm_tokens} ({self.slm_tokens / max(self.total_tokens, 1) * 100:.1f}%)"
        )
        print(
            f"LLM tokens: {self.llm_tokens} ({self.llm_tokens / max(self.total_tokens, 1) * 100:.1f}%)"
        )

        if self.entropy_values:
            entropies = torch.tensor(self.entropy_values)
            print("\nEntropy Distribution:")
            print(f"  Mean: {entropies.mean():.3f} ± {entropies.std():.3f}")
            print(f"  Min: {entropies.min():.3f}, Max: {entropies.max():.3f}")
            print(f"  Median: {entropies.median():.3f}")

            # Percentiles
            p25 = torch.quantile(entropies, 0.25).item()
            p75 = torch.quantile(entropies, 0.75).item()
            p90 = torch.quantile(entropies, 0.90).item()
            print(f"  25th percentile: {p25:.3f}")
            print(f"  75th percentile: {p75:.3f}")
            print(f"  90th percentile: {p90:.3f}")

        if self.uncertainty_values:
            uncertainties = torch.tensor(self.uncertainty_values)
            print("\nUncertainty Distribution:")
            print(f"  Mean: {uncertainties.mean():.3f} ± {uncertainties.std():.3f}")
            print(f"  Min: {uncertainties.min():.3f}, Max: {uncertainties.max():.3f}")
            print(f"  Median: {uncertainties.median():.3f}")

        # Display recent tokens
        if self.recent_tokens:
            print(f"\n{'-' * 70}")
            print(f"Recent Tokens (last {len(self.recent_tokens)}):")
            print(f"{'-' * 70}")

            # Reconstruct text without indicators
            text_parts = [token_info["text"] for token_info in self.recent_tokens]
            reconstructed = "".join(text_parts)
            print(f"{reconstructed}")
            print(f"{'-' * 70}")

        print(f"{'=' * 70}\n")

    def get_final_stats(self) -> dict:
        """Return final statistics dictionary."""
        return {
            "slm_tokens": self.slm_tokens,
            "llm_tokens": self.llm_tokens,
            "total_tokens": self.total_tokens,
            "entropy_values": self.entropy_values,
            "uncertainty_values": self.uncertainty_values,
        }


class GenerationStrategyV0(ABC):
    """Base class for generation strategies."""

    @abstractmethod
    def generate(
        self,
        slm,
        llm,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        device: str,
        **kwargs,
    ) -> tuple[str, dict]:
        """
        Generate text using this strategy.

        Returns:
            tuple of (generated_text, statistics_dict)
        """
        pass


class SpeculativeStrategy(GenerationStrategyV0):
    """
    Speculative decoding strategy.

    SLM generates multiple draft tokens, LLM verifies them in parallel.
    """

    def generate(
        self,
        slm,
        llm,
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
        """Generate using speculative decoding."""
        # Initialize KV caches
        slm_cache = DynamicCache(config=slm.config)
        llm_cache = DynamicCache(config=llm.config)

        def _cache_len(cache: DynamicCache) -> int:
            return cache.get_seq_length() if len(cache) > 0 else 0

        def _crop_cache(cache: DynamicCache, new_len: int):
            if new_len < 0:
                new_len = 0
            cache.crop(new_len)

        # Prefill
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

        # Prefill both models
        _ = slm(
            **inputs,
            past_key_values=slm_cache,
            use_cache=True,
            cache_position=cache_position,
        )
        llm_outputs = llm(
            **inputs,
            past_key_values=llm_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Sample first token from LLM
        token, _ = sample_token(
            llm_outputs.logits[:, -1, :], temperature, top_k, top_p, min_p
        )
        generated_ids = torch.cat([inputs["input_ids"], token], dim=-1)
        offset = prompt_len

        # EOS tokens
        eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        print(tokenizer.decode(token[0], skip_special_tokens=True), end="", flush=True)

        # Statistics
        draft_generated = 0
        draft_accepted = 0
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

        # Main loop
        while generated_ids.shape[1] - prompt_len < max_new_tokens:
            decode_steps += 1
            offset = _cache_len(llm_cache)

            # Generate draft tokens and store their probabilities
            draft_tokens = []
            draft_probs = []
            current_token = token
            current_pos = offset
            for _ in range(num_drafts):
                cache_pos = torch.tensor([current_pos], device=device, dtype=torch.long)
                outputs = slm(
                    input_ids=current_token,
                    past_key_values=slm_cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                next_token, probs = sample_token(
                    outputs.logits[:, -1, :], temperature, top_k, top_p, min_p
                )
                draft_tokens.append(next_token)
                draft_probs.append(probs)
                current_token = next_token
                current_pos += 1
            draft_generated += len(draft_tokens)

            # Verify with LLM
            verify_ids = torch.cat([token] + draft_tokens, dim=-1)
            cache_pos = torch.arange(
                offset, offset + num_drafts + 1, device=device, dtype=torch.long
            )
            llm_outputs = llm(
                input_ids=verify_ids,
                past_key_values=llm_cache,
                use_cache=True,
                cache_position=cache_pos,
            )

            # Accept/reject drafts using standard speculative sampling
            llm_logits = llm_outputs.logits[0]
            accept_count = 0
            accepted_tokens = []

            for i in range(num_drafts):
                _, llm_probs = sample_token(
                    llm_logits[i].unsqueeze(0), temperature, top_k, top_p, min_p
                )
                draft_token_id = draft_tokens[i].item()

                # Get probabilities from both models
                p_target = llm_probs[0, draft_token_id].item()
                q_draft = draft_probs[i][0, draft_token_id].item()

                # Acceptance probability: min(1, p(x_i) / q(x_i))
                acceptance_prob = min(1.0, p_target / (q_draft + 1e-10))

                if torch.rand(1).item() < acceptance_prob:
                    # Accept the draft token
                    accepted_tokens.append(draft_token_id)
                    accept_count += 1
                else:
                    # Reject: sample from adjusted distribution
                    adjusted_probs = torch.clamp(
                        llm_probs[0] - draft_probs[i][0], min=0.0
                    )
                    adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)
                    new_token = torch.multinomial(adjusted_probs, num_samples=1)
                    accepted_tokens.append(new_token.item())
                    llm_tokens += 1  # This token comes from LLM
                    break

            draft_accepted += accept_count
            slm_tokens += accept_count

            # Bonus token if all accepted
            if accept_count == num_drafts:
                _, llm_probs = sample_token(
                    llm_logits[-1].unsqueeze(0), temperature, top_k, top_p, min_p
                )
                bonus_token = torch.multinomial(llm_probs, num_samples=1)
                accepted_tokens.append(bonus_token.item())
                llm_tokens += 1  # Bonus token from LLM

            # Update caches
            desired_cache_len = offset + len(accepted_tokens)
            if accept_count == num_drafts and _cache_len(slm_cache) < desired_cache_len:
                cache_pos = torch.tensor(
                    [_cache_len(slm_cache)], device=device, dtype=torch.long
                )
                slm(
                    input_ids=draft_tokens[-1],
                    past_key_values=slm_cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )

            if _cache_len(llm_cache) > desired_cache_len:
                _crop_cache(llm_cache, desired_cache_len)
            if _cache_len(slm_cache) > desired_cache_len:
                _crop_cache(slm_cache, desired_cache_len)

            offset = desired_cache_len

            # Update generated_ids
            accepted_tensor = torch.tensor(
                accepted_tokens, device=device, dtype=torch.long
            ).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, accepted_tensor], dim=-1)

            # Print progress
            for tok in accepted_tokens:
                print(
                    tokenizer.decode([tok], skip_special_tokens=True),
                    end="",
                    flush=True,
                )

            token = torch.tensor(
                [[accepted_tokens[-1]]], device=device, dtype=torch.long
            )

            if any(tok in eos_token_ids for tok in accepted_tokens):
                break

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Statistics
        stats = {
            "total_tokens": generated_ids.shape[1] - prompt_len,
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "elapsed_time": end_time - start_time,
            "draft_generated": draft_generated,
            "draft_accepted": draft_accepted,
        }

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True), stats


class UncertaintyStrategy(GenerationStrategyV0):
    """
    Uncertainty-based routing strategy.

    Routes to LLM when SLM's aleatoric uncertainty exceeds threshold.
    """

    def generate(
        self,
        slm,
        llm,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        device: str,
        threshold: float = 0.5,
        llm_consecutive_tokens: int = 1,
        verbose: bool = False,
        report_live_metrics: bool = False,
        **kwargs,
    ) -> tuple[str, dict]:
        """Generate using uncertainty-based routing.

        Args:
            llm_consecutive_tokens: Number of consecutive tokens to generate with LLM
                when routing to LLM. Default is 1.
        """
        # Initialize live metrics tracker if requested
        live_tracker = (
            LiveMetricsTracker(report_interval=100) if report_live_metrics else None
        )

        # Initialize KV caches
        slm_cache = DynamicCache(config=slm.config)
        llm_cache = DynamicCache(config=llm.config)

        # Prefill
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

        # Prefill both models and get outputs for first token decision
        slm_prefill_outputs = slm(
            **inputs,
            past_key_values=slm_cache,
            use_cache=True,
            cache_position=cache_position,
        )
        llm_prefill_outputs = llm(
            **inputs,
            past_key_values=llm_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        generated_ids = inputs["input_ids"]
        offset = prompt_len

        eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        # Statistics
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

        # Track cache positions for efficient partial prefill
        slm_cache_pos = prompt_len
        llm_cache_pos = prompt_len

        # Helper function to process a generated token
        def _process_token(next_token, model_used, entropy_val, uncertainty_val):
            nonlocal generated_ids, offset
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            offset += 1

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

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

            return next_token.item() in eos_token_ids

        # Sample first token using prefill outputs
        slm_logits = slm_prefill_outputs.logits[:, -1, :]
        aleatoric_uncertainty, _ = compute_logu(slm_logits)
        token_entropy = calculate_token_entropy(slm_logits, temperature)

        if aleatoric_uncertainty < threshold:
            # Use SLM for first token
            first_token, _ = sample_token(slm_logits, temperature, top_k, top_p, min_p)
            slm_tokens += 1
            slm_cache_pos += 1
            _process_token(first_token, "slm", token_entropy, aleatoric_uncertainty)
        else:
            # Use LLM for first token(s)
            hit_eos = False
            for i in range(llm_consecutive_tokens):
                if i == 0:
                    # First token from prefill outputs
                    llm_logits = llm_prefill_outputs.logits[:, -1, :]
                else:
                    # Subsequent tokens need new forward pass
                    llm_cache_pos_tensor = torch.tensor(
                        [llm_cache_pos], device=device, dtype=torch.long
                    )
                    llm_outputs = llm(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=llm_cache,
                        use_cache=True,
                        cache_position=llm_cache_pos_tensor,
                    )
                    llm_logits = llm_outputs.logits[:, -1, :]

                first_token, _ = sample_token(
                    llm_logits, temperature, top_k, top_p, min_p
                )
                llm_tokens += 1
                llm_cache_pos += 1

                if _process_token(
                    first_token, "llm", token_entropy, aleatoric_uncertainty
                ):
                    hit_eos = True
                    break

        # Main loop
        while generated_ids.shape[1] - prompt_len < max_new_tokens:
            if generated_ids[0, -1].item() in eos_token_ids:
                break

            decode_steps += 1

            # Catch up SLM cache if needed
            if slm_cache_pos < offset:
                catchup_ids = generated_ids[:, slm_cache_pos:offset]
                catchup_cache_pos = torch.arange(
                    slm_cache_pos, offset, device=device, dtype=torch.long
                )
                slm_outputs = slm(
                    input_ids=catchup_ids,
                    past_key_values=slm_cache,
                    use_cache=True,
                    cache_position=catchup_cache_pos,
                )
                slm_cache_pos = offset
                # Use the last token's logits for routing decision
                slm_logits = slm_outputs.logits[:, -1, :]
            else:
                # SLM cache is up to date, do single token forward
                cache_pos = torch.tensor(
                    [slm_cache_pos], device=device, dtype=torch.long
                )
                slm_outputs = slm(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=slm_cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                slm_cache_pos += 1
                slm_logits = slm_outputs.logits[:, -1, :]

            # Calculate uncertainty and entropy
            aleatoric_uncertainty, _ = compute_logu(slm_logits)
            token_entropy = calculate_token_entropy(slm_logits, temperature)

            # Route based on uncertainty
            if aleatoric_uncertainty < threshold:
                # Use SLM
                next_token, _ = sample_token(
                    slm_logits, temperature, top_k, top_p, min_p
                )
                slm_tokens += 1

                if _process_token(
                    next_token, "slm", token_entropy, aleatoric_uncertainty
                ):
                    break
            else:
                # Use LLM - generate multiple consecutive tokens
                # First, catch up LLM cache with partial prefill
                if llm_cache_pos < offset:
                    catchup_ids = generated_ids[:, llm_cache_pos:offset]
                    catchup_cache_pos = torch.arange(
                        llm_cache_pos, offset, device=device, dtype=torch.long
                    )
                    _ = llm(
                        input_ids=catchup_ids,
                        past_key_values=llm_cache,
                        use_cache=True,
                        cache_position=catchup_cache_pos,
                    )
                    llm_cache_pos = offset

                # Generate llm_consecutive_tokens tokens with LLM
                hit_eos = False
                for i in range(llm_consecutive_tokens):
                    if generated_ids.shape[1] - prompt_len >= max_new_tokens:
                        break

                    llm_cache_pos_tensor = torch.tensor(
                        [llm_cache_pos], device=device, dtype=torch.long
                    )
                    llm_outputs = llm(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=llm_cache,
                        use_cache=True,
                        cache_position=llm_cache_pos_tensor,
                    )
                    next_token, _ = sample_token(
                        llm_outputs.logits[:, -1, :], temperature, top_k, top_p, min_p
                    )
                    llm_tokens += 1
                    llm_cache_pos += 1

                    if _process_token(
                        next_token, "llm", token_entropy, aleatoric_uncertainty
                    ):
                        hit_eos = True
                        break

                if hit_eos:
                    break

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Statistics
        stats = {
            "total_tokens": generated_ids.shape[1] - prompt_len,
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "elapsed_time": end_time - start_time,
            "threshold": threshold,
            "llm_consecutive_tokens": llm_consecutive_tokens,
        }

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True), stats


class EntropyStrategy(GenerationStrategyV0):
    """
    Entropy-based routing strategy.

    Routes to LLM when SLM's output entropy exceeds threshold.
    """

    def generate(
        self,
        slm,
        llm,
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
    ) -> tuple[str, dict]:
        """Generate using entropy-based routing."""

        # Initialize live metrics tracker
        live_tracker = (
            LiveMetricsTracker(report_interval=100) if report_live_metrics else None
        )

        # Initialize KV caches
        slm_cache = DynamicCache(config=slm.config)
        llm_cache = DynamicCache(config=llm.config)

        # 1. Prefill Phase
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

        # Prefill both models
        # 注意：这里我们只保留 SLM 的 logits 用于第一次决策
        # LLM 的 prefill 主要是为了填充 KV Cache
        slm_prefill_outputs = slm(
            **inputs,
            past_key_values=slm_cache,
            use_cache=True,
            cache_position=cache_position,
        )
        _ = llm(
            **inputs,
            past_key_values=llm_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        generated_ids = inputs["input_ids"]
        offset = prompt_len  # offset 始终代表当前 generated_ids 的总长度

        # 追踪模型实际已经处理过的 token 数量 (KV Cache 中的长度)
        slm_cache_pos = prompt_len
        llm_cache_pos = prompt_len

        eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        # Statistics
        slm_tokens = 0
        llm_tokens = 0
        decode_steps = 0

        # Helper function
        def _process_token(next_token, model_used, entropy_val, uncertainty_val):
            nonlocal generated_ids, offset
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            offset += 1  # 更新总长度

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
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
            return next_token.item() in eos_token_ids

        # 初始化当前步骤的 logits (来自 prefill)
        current_slm_logits = slm_prefill_outputs.logits[:, -1, :]

        # 2. Main Generation Loop
        while generated_ids.shape[1] - prompt_len < max_new_tokens:
            decode_steps += 1

            # --- Decision Phase ---
            # 基于当前的 SLM logits 计算熵
            entropy = calculate_token_entropy(current_slm_logits, temperature)
            aleatoric_uncertainty, _ = compute_logu(current_slm_logits)

            use_llm = entropy >= threshold

            if not use_llm:
                # === SLM Generation Branch ===
                next_token, _ = sample_token(
                    current_slm_logits, temperature, top_k, top_p, min_p
                )
                slm_tokens += 1

                if _process_token(next_token, "slm", entropy, aleatoric_uncertainty):
                    break

                # 注意：此时生成的 token 还没进入 SLM 的 Cache，将在循环底部的 Catch-up 阶段处理

            else:
                # === LLM Generation Branch ===
                # 策略：如果切换到 LLM，通过 LLM 生成 llm_consecutive_tokens 个 token

                # 1. LLM Catch-up & Get First Logits
                # 如果 LLM 落后了（之前是 SLM 在生成），需要先追平
                llm_next_logits = None

                if llm_cache_pos < offset:
                    catchup_ids = generated_ids[:, llm_cache_pos:offset]
                    catchup_pos = torch.arange(
                        llm_cache_pos, offset, device=device, dtype=torch.long
                    )
                    llm_outputs = llm(
                        input_ids=catchup_ids,
                        past_key_values=llm_cache,
                        use_cache=True,
                        cache_position=catchup_pos,
                    )
                    llm_cache_pos = offset
                    # 优化：直接使用 Catch-up 产生的 logits 预测 LLM 的第一个 token
                    llm_next_logits = llm_outputs.logits[:, -1, :]
                else:
                    # 极端情况：如果刚用完 LLM 又接着用 LLM（通常不会发生，除非 consecutive=1）
                    # 或者如果这是第一步且 prefill 已经完成
                    # 我们需要手动 forward 获取 logits
                    pass  # 逻辑在下面循环处理

                hit_eos = False
                for i in range(llm_consecutive_tokens):
                    if generated_ids.shape[1] - prompt_len >= max_new_tokens:
                        break

                    # 如果没有现成的 logits (即不是刚 Catch-up 完，或者是连续生成的第 2+ 个 token)
                    # 则需要运行 LLM forward
                    if llm_next_logits is None:
                        current_pos = torch.tensor(
                            [llm_cache_pos], device=device, dtype=torch.long
                        )
                        llm_outputs = llm(
                            input_ids=generated_ids[:, -1:],  # 上一步生成的 token
                            past_key_values=llm_cache,
                            use_cache=True,
                            cache_position=current_pos,
                        )
                        llm_cache_pos += 1  # LLM 处理了一个新 token
                        llm_next_logits = llm_outputs.logits[:, -1, :]

                    # 采样
                    next_token, _ = sample_token(
                        llm_next_logits, temperature, top_k, top_p, min_p
                    )
                    llm_tokens += 1

                    # Logits 已消费
                    llm_next_logits = None

                    # 记录 (使用触发切换时的 entropy 值作为参考)
                    if _process_token(
                        next_token, "llm", entropy, aleatoric_uncertainty
                    ):
                        hit_eos = True
                        break

                if hit_eos:
                    break

            # --- Synchronization Phase (SLM Catch-up) ---
            # 无论刚才谁生成了 Token (SLM 生成了 1 个，或者 LLM 生成了 N 个)
            # 我们都需要运行 SLM 来：
            # 1. 更新 SLM 的 KV Cache (让它知道发生了什么)
            # 2. 计算下一步的 Logits (用于下一次循环的熵计算)

            if slm_cache_pos < offset:
                # 获取所有 SLM 未见的 tokens
                new_inputs = generated_ids[:, slm_cache_pos:offset]

                # 严格构造位置索引，确保连续性
                current_cache_position = torch.arange(
                    slm_cache_pos, offset, device=device, dtype=torch.long
                )

                slm_outputs = slm(
                    input_ids=new_inputs,
                    past_key_values=slm_cache,
                    use_cache=True,
                    cache_position=current_cache_position,
                )

                # 更新状态
                slm_cache_pos = offset
                current_slm_logits = slm_outputs.logits[:, -1, :]
            else:
                # 理论上不应到达这里，除非 max_new_tokens 限制导致循环提前结束
                pass

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        # Statistics
        stats = {
            "total_tokens": generated_ids.shape[1] - prompt_len,
            "slm_tokens": slm_tokens,
            "llm_tokens": llm_tokens,
            "decode_steps": decode_steps,
            "elapsed_time": end_time - start_time,
            "threshold": threshold,
            "llm_consecutive_tokens": llm_consecutive_tokens,
        }

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True), stats
