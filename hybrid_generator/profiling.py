"""
Profiling tools for hybrid generation.

This module provides tools to analyze and visualize the generation process,
including uncertainty and entropy distributions for each generated token.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hybrid_generator.strategies import calculate_token_entropy, compute_logu


def _escape_matplotlib_text(text: str) -> str:
    """Escape special characters that matplotlib interprets as LaTeX."""
    # Replace $ with escaped version to prevent LaTeX parsing
    text = text.replace("$", r"\$")
    # Also escape other common LaTeX special characters
    text = text.replace("_", r"\_")
    text = text.replace("^", r"\^")
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    return text


@dataclass
class TokenProfile:
    """Profile information for a single generated token."""

    token_id: int
    token_text: str
    position: int  # Position in the generated sequence
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    entropy: float
    model_used: str  # "slm" or "llm"
    top_k_probs: Optional[List[Tuple[int, float]]] = None  # Top-k token probabilities


@dataclass
class ProfileResult:
    """
    Complete profile result for a generation session.

    Contains token-level statistics and aggregate analysis.
    """

    generated_text: str
    tokens: List[TokenProfile] = field(default_factory=list)
    strategy: str = ""
    total_time: float = 0.0

    # Aggregate statistics
    stats: Dict = field(default_factory=dict)

    def add_token(
        self,
        token_id: int,
        token_text: str,
        position: int,
        aleatoric_uncertainty: float,
        epistemic_uncertainty: float,
        entropy: float,
        model_used: str,
        top_k_probs: Optional[List[Tuple[int, float]]] = None,
    ):
        """Add a token profile to the result."""
        # Convert tensors to Python floats if necessary
        if isinstance(aleatoric_uncertainty, torch.Tensor):
            aleatoric_uncertainty = aleatoric_uncertainty.cpu().item()
        if isinstance(epistemic_uncertainty, torch.Tensor):
            epistemic_uncertainty = epistemic_uncertainty.cpu().item()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.cpu().item()

        self.tokens.append(
            TokenProfile(
                token_id=token_id,
                token_text=token_text,
                position=position,
                aleatoric_uncertainty=aleatoric_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                entropy=entropy,
                model_used=model_used,
                top_k_probs=top_k_probs,
            )
        )

    def get_uncertainties(self, model_filter: Optional[str] = None) -> np.ndarray:
        """Get array of aleatoric uncertainties, optionally filtered by model."""
        tokens = self.tokens
        if model_filter:
            tokens = [t for t in tokens if t.model_used == model_filter]
        # Handle both float and tensor values
        values = []
        for t in tokens:
            val = t.aleatoric_uncertainty
            if isinstance(val, torch.Tensor):
                val = val.cpu().item()
            values.append(val)
        return np.array(values)

    def get_entropies(self, model_filter: Optional[str] = None) -> np.ndarray:
        """Get array of entropies, optionally filtered by model."""
        tokens = self.tokens
        if model_filter:
            tokens = [t for t in tokens if t.model_used == model_filter]
        # Handle both float and tensor values
        values = []
        for t in tokens:
            val = t.entropy
            if isinstance(val, torch.Tensor):
                val = val.cpu().item()
            values.append(val)
        return np.array(values)

    def analyze_distribution(
        self, metric: str = "uncertainty", percentiles: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze the distribution of a metric across generated tokens.

        Args:
            metric: "uncertainty" or "entropy"
            percentiles: List of percentiles to compute (default: [10, 20, 30, 50, 70, 90])

        Returns:
            Dictionary with statistical analysis
        """
        if percentiles is None:
            percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        if metric == "uncertainty":
            values = self.get_uncertainties()
        elif metric == "entropy":
            values = self.get_entropies()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if len(values) == 0:
            return {"error": "No values to analyze"}

        # Sort values from high to low
        sorted_values = np.sort(values)[::-1]

        analysis = {
            "metric": metric,
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "percentiles": {},
            "top_10_percent": {
                "mean": float(np.mean(sorted_values[: max(1, len(values) // 10)])),
                "min": float(sorted_values[0]),
                "max": float(
                    sorted_values[min(len(values) - 1, len(values) // 10 - 1)]
                ),
            },
        }

        # Compute percentiles (from high to low)
        for p in percentiles:
            # p-th percentile from the top
            idx = int(len(sorted_values) * p / 100)
            if idx < len(sorted_values):
                analysis["percentiles"][f"top_{p}%"] = float(sorted_values[idx])

        return analysis

    def get_high_uncertainty_tokens(
        self, top_n: int = 10, metric: str = "uncertainty"
    ) -> List[TokenProfile]:
        """
        Get tokens with highest uncertainty/entropy.

        Args:
            top_n: Number of top tokens to return
            metric: "uncertainty" or "entropy"

        Returns:
            List of TokenProfile sorted by metric (descending)
        """

        def _get_value(val):
            """Extract float value from tensor or float."""
            if isinstance(val, torch.Tensor):
                return val.cpu().item()
            return val

        if metric == "uncertainty":
            sorted_tokens = sorted(
                self.tokens,
                key=lambda t: _get_value(t.aleatoric_uncertainty),
                reverse=True,
            )
        elif metric == "entropy":
            sorted_tokens = sorted(
                self.tokens, key=lambda t: _get_value(t.entropy), reverse=True
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted_tokens[:top_n]

    def print_summary(self):
        """Print a comprehensive summary of the profile."""
        print("\n" + "=" * 70)
        print(f"Profile Summary - Strategy: {self.strategy}")
        print("=" * 70)

        # Routing configuration (if available)
        if "enable_routing" in self.stats:
            print("\nProfiling Configuration:")
            if self.stats["enable_routing"]:
                print("  Routing: Enabled")
                print(f"  Routing Metric: {self.stats.get('routing_metric', 'N/A')}")
                print(f"  Threshold: {self.stats.get('threshold', 'N/A')}")
            else:
                print("  Routing: Disabled (SLM only)")
                print(
                    "  Note: All tokens generated by SLM to profile complete distribution"
                )

        # Basic stats
        print(f"\nGenerated {len(self.tokens)} tokens in {self.total_time:.2f}s")
        print(f"Speed: {len(self.tokens) / max(self.total_time, 1e-9):.2f} tok/s")

        # Model usage
        slm_tokens = sum(1 for t in self.tokens if t.model_used == "slm")
        llm_tokens = sum(1 for t in self.tokens if t.model_used == "llm")
        if slm_tokens + llm_tokens > 0:
            print("\nModel Usage:")
            print(
                f"  SLM: {slm_tokens} tokens ({slm_tokens / (slm_tokens + llm_tokens) * 100:.1f}%)"
            )
            print(
                f"  LLM: {llm_tokens} tokens ({llm_tokens / (slm_tokens + llm_tokens) * 100:.1f}%)"
            )

        # Uncertainty analysis
        print("\n" + "-" * 70)
        print("Uncertainty Distribution:")
        print("-" * 70)
        uncertainty_analysis = self.analyze_distribution("uncertainty")
        self._print_distribution_analysis(uncertainty_analysis)

        # Entropy analysis
        print("\n" + "-" * 70)
        print("Entropy Distribution:")
        print("-" * 70)
        entropy_analysis = self.analyze_distribution("entropy")
        self._print_distribution_analysis(entropy_analysis)

        # High uncertainty tokens
        print("\n" + "-" * 70)
        print("Top 10 Most Uncertain Tokens:")
        print("-" * 70)
        high_uncertainty = self.get_high_uncertainty_tokens(10, "uncertainty")
        for i, token in enumerate(high_uncertainty, 1):
            # Extract float values for printing
            uncertainty = token.aleatoric_uncertainty
            entropy = token.entropy
            if isinstance(uncertainty, torch.Tensor):
                uncertainty = uncertainty.cpu().item()
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.cpu().item()

            print(
                f"{i:2d}. Pos {token.position:3d} | "
                f'"{token.token_text:20s}" | '
                f"Uncertainty: {uncertainty:.4f} | "
                f"Entropy: {entropy:.4f} | "
                f"Model: {token.model_used.upper()}"
            )

        print("\n" + "=" * 70 + "\n")

    def _print_distribution_analysis(self, analysis: Dict):
        """Helper to print distribution analysis."""
        if "error" in analysis:
            print(f"  {analysis['error']}")
            return

        print(f"  Count: {analysis['count']}")
        print(f"  Mean: {analysis['mean']:.4f} ± {analysis['std']:.4f} (std)")
        print(f"  Range: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        print(f"  Median: {analysis['median']:.4f}")

        print("\n  Top 10% tokens:")
        print(
            f"    Mean: {analysis['top_10_percent']['mean']:.4f}, "
            f"Range: [{analysis['top_10_percent']['min']:.4f}, {analysis['top_10_percent']['max']:.4f}]"
        )

        print("\n  Percentiles (from highest):")
        for percentile, value in sorted(analysis["percentiles"].items()):
            print(f"    {percentile}: {value:.4f}")

    def save_to_file(self, filename: str):
        """Save profile data to a JSON file."""
        import json

        def _to_float(val):
            """Convert tensor or float to Python float."""
            if isinstance(val, torch.Tensor):
                return val.cpu().item()
            return float(val)

        data = {
            "generated_text": self.generated_text,
            "strategy": self.strategy,
            "total_time": self.total_time,
            "stats": self.stats,
            "tokens": [
                {
                    "token_id": t.token_id,
                    "token_text": t.token_text,
                    "position": t.position,
                    "aleatoric_uncertainty": _to_float(t.aleatoric_uncertainty),
                    "epistemic_uncertainty": _to_float(t.epistemic_uncertainty),
                    "entropy": _to_float(t.entropy),
                    "model_used": t.model_used,
                }
                for t in self.tokens
            ],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Profile saved to: {filename}")

    def plot_distributions(self, save_path: Optional[str] = None):
        """
        Plot uncertainty and entropy distributions.

        Args:
            save_path: If provided, save the plot to this path instead of showing
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        uncertainties = self.get_uncertainties()
        entropies = self.get_entropies()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Uncertainty distribution
        axes[0, 0].hist(uncertainties, bins=50, alpha=0.7, color="blue")
        axes[0, 0].set_xlabel("Aleatoric Uncertainty")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Uncertainty Distribution")
        axes[0, 0].grid(True, alpha=0.3)

        # Entropy distribution
        axes[0, 1].hist(entropies, bins=50, alpha=0.7, color="green")
        axes[0, 1].set_xlabel("Entropy")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Entropy Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # Uncertainty over position
        positions = [t.position for t in self.tokens]
        axes[1, 0].scatter(positions, uncertainties, alpha=0.5, s=10)
        axes[1, 0].set_xlabel("Token Position")
        axes[1, 0].set_ylabel("Aleatoric Uncertainty")
        axes[1, 0].set_title("Uncertainty Over Generation")
        axes[1, 0].grid(True, alpha=0.3)

        # Entropy over position
        axes[1, 1].scatter(positions, entropies, alpha=0.5, s=10, color="green")
        axes[1, 1].set_xlabel("Token Position")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].set_title("Entropy Over Generation")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"Generation Profile - Strategy: {self.strategy}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_token_sequence(
        self,
        save_path: str = "token_sequence.pdf",
        metric: str = "entropy",
        tokens_per_line: int = 10,
        figsize_per_token: tuple = (1, 0.6),
        highlight_tokens: list = None,
    ):
        """
        Visualize the token sequence with color-coding based on entropy or uncertainty.

        Creates a flame-graph-like visualization where each token is colored according to
        its entropy or uncertainty value. Higher values use warmer colors (red), lower
        values use cooler colors (blue).

        Args:
            save_path: Path to save the PDF file
            metric: "entropy" or "uncertainty" - which metric to use for coloring
            tokens_per_line: Number of tokens to display per line
            figsize_per_token: (width, height) size per token for figure sizing
            highlight_tokens: List of token texts to highlight with special color (e.g., ["<think>", "</think>"])
        """
        if highlight_tokens is None:
            highlight_tokens = ["<think>", "</think>"]
        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        if len(self.tokens) == 0:
            print("No tokens to visualize")
            return

        # Get metric values
        if metric == "entropy":
            values = self.get_entropies()
            metric_label = "Entropy"
        elif metric == "uncertainty":
            values = self.get_uncertainties()
            metric_label = "Aleatoric Uncertainty"
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Use 'entropy' or 'uncertainty'"
            )

        # Normalize values for color mapping
        vmin, vmax = values.min(), values.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("YlOrRd")  # Yellow-Orange-Red colormap (flame-like)

        # Calculate figure size based on token count
        num_lines = (len(self.tokens) + tokens_per_line - 1) // tokens_per_line
        fig_width = tokens_per_line * figsize_per_token[0]
        fig_height = num_lines * figsize_per_token[1] + 2  # +2 for colorbar space

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, tokens_per_line)
        ax.set_ylim(0, num_lines + 1)
        ax.axis("off")

        # Add title
        title = f"Token Sequence Visualization - {metric_label}"
        if not self.stats.get("enable_routing", True):
            title += " (SLM Only)"
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        # Draw each token as a colored box with text
        for idx, (token, value) in enumerate(zip(self.tokens, values)):
            line = idx // tokens_per_line
            col = idx % tokens_per_line

            # Calculate position (inverted y-axis so first line is at top)
            x = col
            y = num_lines - line - 0.5

            # Check if this token should be highlighted
            is_highlighted = token.token_text.strip() in highlight_tokens

            # Get color based on metric value
            if is_highlighted:
                # Use a distinctive color for highlighted tokens (cyan/turquoise)
                color = "#00CED1"  # Dark Turquoise
                edge_color = "#FF1493"  # Deep Pink for extra visibility
                edge_width = 3.0
            else:
                color = cmap(norm(value))
                edge_color = "black"
                edge_width = 0.5

            # Create rectangle for token
            rect = mpatches.Rectangle(
                (x, y - 0.4),
                0.95,
                0.8,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
            )
            ax.add_patch(rect)

            # Add token text
            token_text = token.token_text
            # Truncate if too long
            if len(token_text) > 8:
                token_text = token_text[:7] + "..."
            # Escape special characters for matplotlib
            token_text = _escape_matplotlib_text(token_text)

            # Determine text color based on background
            if is_highlighted:
                text_color = "black"  # Dark text on cyan background
            else:
                text_color = "black" if norm(value) < 0.5 else "white"

            ax.text(
                x + 0.475,
                y,
                token_text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_color,
                fontfamily="monospace",
            )

            # Add metric value below token (small text) - only for non-highlighted tokens
            if not is_highlighted:
                ax.text(
                    x + 0.475,
                    y - 0.3,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                    fontfamily="monospace",
                )

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, orientation="horizontal", pad=0.02, aspect=40, shrink=0.8
        )
        cbar.set_label(metric_label, fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

        # Add statistics text
        stats_text = (
            f"Total Tokens: {len(self.tokens)} | "
            f"Mean {metric_label}: {values.mean():.4f} | "
            f"Range: [{vmin:.4f}, {vmax:.4f}]"
        )

        # Check if any highlighted tokens exist in the sequence
        has_highlighted = any(
            t.token_text.strip() in highlight_tokens for t in self.tokens
        )

        if has_highlighted:
            escaped_highlights = [_escape_matplotlib_text(t) for t in highlight_tokens]
            highlight_text = f" | Highlighted: {', '.join(escaped_highlights)}"
            stats_text += highlight_text

        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Token sequence visualization saved to: {save_path}")
        plt.close()


def profile_token_generation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    position: int,
    model_name: str,
    temperature: float = 1.0,
    top_k_size: int = 5,
) -> Tuple[float, float, float, Optional[List[Tuple[int, float]]]]:
    """
    Profile a single token generation step.

    Args:
        model: The model being used
        tokenizer: The tokenizer
        input_ids: Current input token IDs
        logits: Model output logits for next token
        position: Position in sequence
        model_name: "slm" or "llm"
        temperature: Temperature for entropy calculation
        top_k_size: Number of top probabilities to record

    Returns:
        Tuple of (aleatoric_uncertainty, epistemic_uncertainty, entropy, top_k_probs)
    """
    # Calculate uncertainties
    aleatoric_uncertainty, epistemic_uncertainty = compute_logu(logits)

    # Calculate entropy
    entropy = calculate_token_entropy(logits, temperature)
    if isinstance(entropy, torch.Tensor):
        entropy = entropy.item()

    # Get top-k probabilities
    probs = torch.softmax(logits / temperature, dim=-1)
    top_k_probs_tensor, top_k_indices = torch.topk(probs, top_k_size)

    top_k_probs = [
        (int(top_k_indices[i]), float(top_k_probs_tensor[i])) for i in range(top_k_size)
    ]

    return aleatoric_uncertainty, epistemic_uncertainty, entropy, top_k_probs
