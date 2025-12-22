"""
LM-Eval wrapper for HybridGenerator.

This module provides a wrapper class that makes HybridGenerator compatible
with the lm-evaluation-harness framework for model evaluation.
"""

import time
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Support both relative and absolute imports
try:
    from .generator import HybridGenerator
except ImportError:
    from generator import HybridGenerator


@register_model("hybrid")
class HybridLM(LM):
    """
    Wrapper class that makes HybridGenerator compatible with lm-evaluation-harness.

    This class implements the required interface methods for lm-eval:
    - loglikelihood(): Compute log-likelihood for completion given context
    - generate_until(): Generate text until stopping condition
    - loglikelihood_rolling(): Compute rolling log-likelihood

    Example:
        # Register and use with lm-eval CLI
        $ lm_eval --model hybrid \\
                  --model_args slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty \\
                  --tasks hellaswag,arc_easy \\
                  --device cuda \\
                  --batch_size 1

        # Or use programmatically
        >>> from lm_eval import evaluator
        >>> model = HybridLM(
        ...     slm_model_id="Qwen/Qwen3-1.7B",
        ...     llm_model_id="Qwen/Qwen3-8B",
        ...     strategy="uncertainty"
        ... )
        >>> results = evaluator.simple_evaluate(
        ...     model=model,
        ...     tasks=["hellaswag"],
        ...     batch_size=1
        ... )
    """

    def __init__(
        self,
        slm_model_id: str,
        llm_model_id: str,
        strategy: Literal["speculative", "uncertainty", "entropy"] = "uncertainty",
        threshold: float = 0.5,
        num_drafts: int = 4,
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 1,
        use_slm_only: bool = False,
        report_routing_metrics: bool = False,
        report_content: bool = False,
        **kwargs,
    ):
        """
        Initialize HybridLM for lm-evaluation.

        Args:
            slm_model_id: HuggingFace model ID for small/fast model
            llm_model_id: HuggingFace model ID for large/accurate model
            strategy: Generation strategy ("speculative", "uncertainty", "entropy")
            threshold: Threshold for uncertainty/entropy routing
            num_drafts: Number of drafts for speculative decoding
            device: Device to run on
            dtype: Data type ("float16", "bfloat16", "float32")
            batch_size: Batch size for evaluation (currently only supports 1)
            use_slm_only: If True, use only SLM for all operations and skip loading LLM
                weights (default: False)
            report_routing_metrics: If True, print token routing breakdown (SLM vs LLM) after evaluation
            report_content: If True, record and report detailed content for each question
            **kwargs: Additional arguments
        """
        super().__init__()

        self.slm_model_id = slm_model_id
        self.llm_model_id = llm_model_id
        self.report_routing_metrics_flag = report_routing_metrics
        self.report_content_flag = report_content
        # Convert dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        self.device = device

        if use_slm_only:
            # Skip loading the LLM entirely for SLM-only baselines
            tokenizer = AutoTokenizer.from_pretrained(slm_model_id)
            slm = AutoModelForCausalLM.from_pretrained(
                slm_model_id, torch_dtype=torch_dtype
            ).to(device)
            slm.eval()
            self.generator = SimpleNamespace(
                slm=slm,
                llm=None,
                tokenizer=tokenizer,
            )
        else:
            # Initialize HybridGenerator with both SLM and LLM
            # Set verbose=False to avoid printing tokens during evaluation
            # Enable live metrics reporting if report_routing_metrics is True
            self.generator = HybridGenerator(
                slm_model_id=slm_model_id,
                llm_model_id=llm_model_id,
                device=device,
                dtype=torch_dtype,
                verbose=False,
                report_live_metrics=report_routing_metrics,
            )

        self.strategy = strategy
        self.threshold = threshold
        self.num_drafts = num_drafts
        self._batch_size = batch_size
        self.use_slm_only = use_slm_only
        self._metrics_total: Dict[str, Any] = {
            "slm_tokens": 0,
            "llm_tokens": 0,
            "total_tokens": 0,
            "decode_steps": 0,
        }
        self._detailed_content: List[Dict[str, Any]] = []
        self._total_generation_time: float = 0.0

        # Model attributes required by lm-eval
        self.tokenizer = self.generator.tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def batch_size(self):
        """Batch size for evaluation."""
        return self._batch_size

    @property
    def eot_token_id(self):
        """End of text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """Maximum sequence length."""
        # Use the smaller model's max length as the limit
        slm_max = self.generator.slm.config.max_position_embeddings
        llm = getattr(self.generator, "llm", None)
        if llm is None:
            return slm_max
        return min(slm_max, llm.config.max_position_embeddings)

    @property
    def max_gen_toks(self):
        """Maximum number of tokens to generate."""
        return 32768

    @property
    def tokenizer_name(self):
        """Name of the tokenizer (required for chat template support)."""
        # Return the model ID that was used to load the tokenizer
        if self.use_slm_only:
            return self.slm_model_id
        else:
            return self.llm_model_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Encode string to token IDs."""
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def apply_chat_template(self, chat_history: List[Dict[str, str]], **kwargs) -> str:
        """
        Apply chat template to format messages.

        Args:
            chat_history: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to tokenizer.apply_chat_template

        Returns:
            Formatted string with chat template applied
        """
        # Set default value for tokenize if not provided
        if "tokenize" not in kwargs:
            kwargs["tokenize"] = False

        # Pass through all other arguments to the tokenizer's apply_chat_template
        return self.tokenizer.apply_chat_template(chat_history, **kwargs)

    def _extract_args(self, request):
        """
        Normalize request arguments from lm-eval.

        Newer lm-eval versions pass Instance objects with an .args attribute,
        while older versions pass raw tuples. This helper keeps both working.
        """
        args = getattr(request, "args", request)
        if not isinstance(args, (list, tuple)):
            raise TypeError(f"Unexpected request type: {type(request)}")
        return args

    @torch.inference_mode()
    def _model_call(self, inputs: torch.Tensor, use_slm: bool = False) -> torch.Tensor:
        """
        Internal method to get logits from either SLM or LLM.

        Args:
            inputs: Input token IDs [batch_size, seq_len]
            use_slm: Whether to use SLM (default: use LLM)

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        llm_available = getattr(self.generator, "llm", None) is not None
        # Fall back to SLM if LLM is not loaded
        model = (
            self.generator.slm if use_slm or not llm_available else self.generator.llm
        )
        outputs = model(inputs)
        return outputs.logits

    @torch.inference_mode()
    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.

        This is the core method used by most lm-eval tasks for multiple-choice
        and other likelihood-based evaluations.

        Args:
            requests: List of (context, continuation) tuples

        Returns:
            List of (log_likelihood, is_greedy) tuples where:
            - log_likelihood: sum of log probabilities for continuation tokens
            - is_greedy: whether continuation matches greedy decoding
        """
        results = []

        for req in requests:
            context, continuation = self._extract_args(req)
            # Tokenize context and continuation
            context_ids = self.tok_encode(context)
            continuation_ids = self.tok_encode(continuation)

            # Combine context and continuation
            input_ids = torch.tensor(
                [context_ids + continuation_ids], device=self.device
            )

            # Get logits from the appropriate model
            # If use_slm_only is True, use SLM; otherwise use LLM (more accurate)
            logits = self._model_call(input_ids, use_slm=self.use_slm_only)

            # Calculate log probabilities for continuation tokens
            # logits shape: [1, seq_len, vocab_size]
            # We want log probs for tokens at positions [len(context):len(context)+len(continuation)]
            context_len = len(context_ids)
            continuation_len = len(continuation_ids)

            # Get logits for positions that predict continuation tokens
            continuation_logits = logits[
                0, context_len - 1 : context_len + continuation_len - 1, :
            ]

            # Convert to log probabilities
            log_probs = F.log_softmax(continuation_logits, dim=-1)

            # Get log probability of actual continuation tokens
            continuation_tensor = torch.tensor(continuation_ids, device=self.device)
            token_log_probs = log_probs[range(continuation_len), continuation_tensor]

            # Sum log probabilities
            total_log_prob = token_log_probs.sum().item()

            # Check if greedy decoding would produce the same continuation
            greedy_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = torch.all(greedy_tokens == continuation_tensor).item()

            results.append((total_log_prob, is_greedy))

        return results

    @torch.inference_mode()
    def loglikelihood_rolling(self, requests: List[Tuple[str,]]) -> List[Tuple[float,]]:
        """
        Compute rolling log-likelihood for sequences.

        This method is used for perplexity evaluation and similar metrics.

        Args:
            requests: List of (text,) tuples

        Returns:
            List of (log_likelihood,) tuples
        """
        results = []

        for req in requests:
            (text,) = self._extract_args(req)
            # Tokenize the text
            token_ids = self.tok_encode(text)

            if len(token_ids) <= 1:
                results.append((0.0,))
                continue

            # Create input tensor
            input_ids = torch.tensor([token_ids], device=self.device)

            # Get logits from the appropriate model
            logits = self._model_call(
                input_ids,
                use_slm=self.use_slm_only,
            )

            # Calculate log probabilities
            log_probs = F.log_softmax(logits[0], dim=-1)

            # Sum log probabilities for all tokens (rolling)
            # For each position i, get log prob of token i given tokens 0..i-1
            token_log_probs = log_probs[:-1, :][
                range(len(token_ids) - 1), token_ids[1:]
            ]
            total_log_prob = token_log_probs.sum().item()

            results.append((total_log_prob,))

        return results

    @torch.inference_mode()
    def generate_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        """
        Generate text until stopping condition is met.

        This method is used for generation-based tasks like text completion.

        Args:
            requests: List of (context, generation_kwargs) tuples where
                     generation_kwargs may contain:
                     - until: List of stop strings
                     - max_gen_toks: Maximum tokens to generate
                     - temperature: Sampling temperature
                     - do_sample: Whether to use sampling

        Returns:
            List of generated strings
        """
        results = []
        print("TODO: Generation temperature: 0.6")
        print(f"number of requests: {len(requests)}")
        for req in requests:
            print(f"Processing request: {req}")
            start_time = time.time()
            context, gen_kwargs = self._extract_args(req)
            # Extract generation parameters
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            # temperature = gen_kwargs.get("temperature", 0.6)

            do_sample = gen_kwargs.get("do_sample", False)

            temperature = 0.6
            do_sample = True

            # Adjust temperature for sampling
            # if temperature == 0.0:
            #     temperature = 1.0
            #     do_sample = False

            can_use_strategy = (
                hasattr(self.generator, "generate")
                and not self.use_slm_only
                and getattr(self.generator, "llm", None) is not None
            )

            if can_use_strategy:
                generated = self._generate_with_strategy(
                    context=context,
                    max_new_tokens=max_gen_toks,
                    temperature=temperature,
                    do_sample=do_sample,
                    stop_strings=until,
                )
            else:
                # Fall back to simple generation (e.g., SLM-only runs)
                generated = self._generate_simple(
                    context=context,
                    max_new_tokens=max_gen_toks,
                    temperature=temperature,
                    do_sample=do_sample,
                    stop_strings=until,
                )

            results.append(generated)

            # Record detailed content if enabled
            if self.report_content_flag:
                end_time = time.time()
                generation_time = end_time - start_time
                num_tokens = len(self.tok_encode(generated))

                self._detailed_content.append(
                    {
                        "question": context,
                        "answer": generated,
                        "num_tokens": num_tokens,
                        "generation_time": generation_time,
                    }
                )
                self._total_generation_time += generation_time

        return results

    def _accumulate_metrics(self, stats: Optional[Dict[str, Any]]):
        """Accumulate stats returned from HybridGenerator.generate."""
        if not stats:
            return
        for key in ["slm_tokens", "llm_tokens", "total_tokens", "decode_steps"]:
            if key in stats and stats[key] is not None:
                self._metrics_total[key] += int(stats[key])

    def _generate_simple(
        self,
        context: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stop_strings: List[str],
    ) -> str:
        """
        Simple generation method for lm-eval compatibility.

        Uses the appropriate model (SLM or LLM) based on use_slm_only parameter.
        """
        # Select model based on use_slm_only and availability
        llm_available = getattr(self.generator, "llm", None) is not None
        use_slm = self.use_slm_only or not llm_available
        model_name = "slm" if use_slm else "llm"
        model = self.generator.slm if use_slm else self.generator.llm

        # Tokenize context
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Initialize cache
        cache = DynamicCache(config=model.config)
        offset = input_ids.shape[1]

        # Prefill
        cache_position = torch.arange(offset, device=self.device, dtype=torch.long)
        outputs = model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        generated_ids = input_ids
        generated_text = context

        # Generation loop
        for _ in range(max_new_tokens):
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                # Sample with temperature
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Decode and check for stop strings
            new_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Check if any stop string is in the generated text
            should_stop = False
            for stop_str in stop_strings:
                if stop_str in new_text[len(context) :]:
                    # Truncate at stop string
                    stop_idx = new_text.index(stop_str, len(context))
                    generated_text = new_text[:stop_idx]
                    should_stop = True
                    break

            if should_stop:
                break

            generated_text = new_text

            # Continue generation
            cache_pos = torch.tensor([offset], device=self.device, dtype=torch.long)
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos,
            )
            offset += 1

        # Return only the generated part (exclude context)
        return generated_text[len(context) :]

    def _generate_with_strategy(
        self,
        context: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stop_strings: List[str],
    ) -> str:
        """
        Generate using HybridGenerator strategies to collect built-in metrics.
        """
        temp = temperature if do_sample else 0.0

        generated, stats = self.generator.generate(
            prompt=context,
            strategy=self.strategy,
            max_new_tokens=max_new_tokens,
            temperature=temp if temp > 0 else 0.0,
            top_k=20,
            top_p=0.95,
            min_p=0.0,
            num_drafts=self.num_drafts,
            threshold=self.threshold,
        )

        self._accumulate_metrics(stats)

        completion = generated[len(context) :]
        for stop_str in stop_strings:
            if stop_str in completion:
                stop_idx = completion.index(stop_str)
                completion = completion[:stop_idx]
                break
        return completion

    def __repr__(self):
        """String representation of the model."""
        slm_name = self.generator.slm.config.name_or_path
        llm = getattr(self.generator, "llm", None)
        llm_name = (
            llm.config.name_or_path
            if llm is not None
            else f"{self.llm_model_id} (not loaded)"
        )
        model_info = (
            f"HybridLM(slm={slm_name}, llm={llm_name}, strategy={self.strategy}"
        )
        if self.use_slm_only:
            model_info += ", use_slm_only=True"
        model_info += ")"
        return model_info

    def get_routing_metrics(self) -> Dict[str, int]:
        """Return accumulated routing metrics (SLM vs LLM tokens) sourced from HybridGenerator."""
        return {
            "slm_tokens": self._metrics_total.get("slm_tokens", 0),
            "llm_tokens": self._metrics_total.get("llm_tokens", 0),
            "total_tokens": self._metrics_total.get("total_tokens", 0),
            "decode_steps": self._metrics_total.get("decode_steps", 0),
        }

    def get_content_metrics(self) -> Dict[str, Any]:
        """Return detailed content metrics for each question including text, token counts, and timing."""
        return {
            "detailed_content": self._detailed_content,
            "total_generation_time": self._total_generation_time,
            "total_questions": len(self._detailed_content),
        }

    def report_routing_metrics(self):
        """Print token routing metrics (SLM vs LLM usage)."""
        if not self.report_routing_metrics_flag:
            return

        metrics = self.get_routing_metrics()
        slm_tokens = metrics["slm_tokens"]
        llm_tokens = metrics["llm_tokens"]
        total_tokens = metrics["total_tokens"]
        decode_steps = metrics["decode_steps"]

        print("\n" + "=" * 70)
        print("Token Routing Metrics")
        print("=" * 70)
        print(f"Tokens - SLM: {slm_tokens}, LLM: {llm_tokens}, Total: {total_tokens}")
        print(f"Decode steps (sum): {decode_steps}")

        if total_tokens > 0:
            slm_ratio = slm_tokens / total_tokens
            print(f"SLM token ratio: {slm_ratio:.2%}")
        else:
            print(
                "No strategy-based metrics collected (generate_until may not have run)."
            )

        print("=" * 70 + "\n")

    def report_content(self):
        """Print detailed content metrics for each question."""
        if not self.report_content_flag:
            return

        if not self._detailed_content:
            print("\n" + "=" * 70)
            print("Content Metrics")
            print("=" * 70)
            print("No content recorded (generate_until may not have run).")
            print("=" * 70 + "\n")
            return

        print("\n" + "=" * 70)
        print("Content Metrics")
        print("=" * 70)
        print(f"Total questions: {len(self._detailed_content)}")
        print(f"Total generation time: {self._total_generation_time:.4f}s")
        if len(self._detailed_content) > 0:
            avg_time = self._total_generation_time / len(self._detailed_content)
            print(f"Average time per question: {avg_time:.4f}s")

        print("\n" + "-" * 70)
        print("Per-Question Details")
        print("-" * 70)

        for i, item in enumerate(self._detailed_content, 1):
            print(f"\nQuestion {i}:")
            print(f"  Tokens generated: {item['num_tokens']}")
            print(f"  Generation time: {item['generation_time']:.4f}s")
            print(
                f"  Question: {item['question'][:100]}..."
                if len(item["question"]) > 100
                else f"  Question: {item['question']}"
            )
            print(
                f"  Answer: {item['answer'][:100]}..."
                if len(item["answer"]) > 100
                else f"  Answer: {item['answer']}"
            )

        print("=" * 70 + "\n")


def create_hybrid_lm_from_args(model_args: str) -> HybridLM:
    """
    Create HybridLM from comma-separated argument string.

    This function is used by lm-eval CLI when parsing --model_args.

    Args:
        model_args: Comma-separated string like
                   "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty,threshold=0.5,use_slm_only=false"

    Returns:
        Initialized HybridLM instance

    Examples:
        # Use LLM for evaluation (default)
        $ lm_eval --model hybrid \\
                  --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,strategy=uncertainty" \\
                  --tasks hellaswag

        # Use only SLM for evaluation (baseline comparison)
        $ lm_eval --model hybrid \\
                  --model_args "slm=Qwen/Qwen3-1.7B,llm=Qwen/Qwen3-8B,use_slm_only=true" \\
                  --tasks hellaswag
    """
    # Parse comma-separated arguments
    args_dict = {}
    for arg in model_args.split(","):
        if "=" in arg:
            key, value = arg.split("=", 1)
            args_dict[key.strip()] = value.strip()

    # Extract required arguments
    slm_model_id = args_dict.get("slm", args_dict.get("slm_model_id"))
    llm_model_id = args_dict.get("llm", args_dict.get("llm_model_id"))

    if not slm_model_id or not llm_model_id:
        raise ValueError(
            "Both 'slm' and 'llm' model IDs must be specified in model_args"
        )

    # Extract optional arguments with defaults
    strategy = args_dict.get("strategy", "uncertainty")
    threshold = float(args_dict.get("threshold", "0.5"))
    num_drafts = int(args_dict.get("num_drafts", "4"))
    device = args_dict.get("device", "cuda")
    dtype = args_dict.get("dtype", "float16")
    batch_size = int(args_dict.get("batch_size", "1"))

    # Parse use_slm_only (support true/false, True/False, 1/0)
    use_slm_only_str = args_dict.get("use_slm_only", "false").lower()
    use_slm_only = use_slm_only_str in ("true", "1", "yes")
    report_routing_metrics_str = args_dict.get(
        "report_routing_metrics", "false"
    ).lower()
    report_routing_metrics = report_routing_metrics_str in ("true", "1", "yes")
    report_content_str = args_dict.get("report_content", "false").lower()
    report_content = report_content_str in ("true", "1", "yes")

    return HybridLM(
        slm_model_id=slm_model_id,
        llm_model_id=llm_model_id,
        strategy=strategy,
        threshold=threshold,
        num_drafts=num_drafts,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        use_slm_only=use_slm_only,
        report_routing_metrics=report_routing_metrics,
        report_content=report_content,
    )
