"""
Hybrid Generator Module

A unified interface for combining small and large language models
using multiple generation strategies.

Main Components:
    - HybridGenerator: Main class for hybrid generation
    - Strategies: SpeculativeStrategy, UncertaintyStrategy, EntropyStrategy
    - Utilities: sample_token, compute_logu, calculate_token_entropy

Example:
    >>> from hybrid_generator import HybridGenerator
    >>>
    >>> generator = HybridGenerator(
    ...     slm_model_id="Qwen/Qwen3-1.7B",
    ...     llm_model_id="Qwen/Qwen3-8B"
    ... )
    >>>
    >>> # Use speculative decoding
    >>> result = generator.generate(
    ...     prompt="Write a story about AI",
    ...     strategy="speculative",
    ...     num_drafts=4
    ... )
    >>>
    >>> # Use uncertainty-based routing
    >>> result = generator.generate(
    ...     prompt="Explain quantum computing",
    ...     strategy="uncertainty",
    ...     threshold=0.5
    ... )
"""

from .generator import HybridGenerator

# Try to import HybridLM (requires lm-eval)
from .lm_eval_wrapper import HybridLM, create_hybrid_lm_from_args
from .profiling import (
    ProfileResult,
    TokenProfile,
    profile_token_generation,
)
from .strategies import (
    EntropyStrategy,
    GenerationStrategy,
    SpeculativeStrategy,
    # UncertaintyStrategy,
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    "HybridGenerator",
    # Strategies
    "GenerationStrategy",
    "SpeculativeStrategy",
    "UncertaintyStrategy",
    "EntropyStrategy",
    # Utilities
    "sample_token",
    "compute_logu",
    "calculate_token_entropy",
    # Profiling
    "ProfileResult",
    "TokenProfile",
    "profile_token_generation",
    # LM-Eval wrapper (optional)
    "HybridLM",
    "create_hybrid_lm_from_args",
]
