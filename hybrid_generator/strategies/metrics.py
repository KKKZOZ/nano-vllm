import torch


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
