from abc import ABC, abstractmethod

from hybrid_generator.backends import ModelBackend


class GenerationStrategy(ABC):
    """Base class for generation strategies."""

    @abstractmethod
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
        **kwargs,
    ) -> tuple[str, dict]:
        """
        Generate text using this strategy.

        Returns:
            tuple of (generated_text, statistics_dict)
        """
        pass
