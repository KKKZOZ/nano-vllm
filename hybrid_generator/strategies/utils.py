"""
Utility functions for hybrid generation.

This module contains helper functions for:
- Sampling (temperature, top-k, top-p, min-p)
- Uncertainty calculation (aleatoric and epistemic)
- Entropy calculation
"""

from typing import Tuple, Union

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a token from logits with temperature, top-k, top-p, and min-p filtering.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        temperature: Temperature for sampling (higher = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Nucleus sampling - keep tokens with cumulative probability <= top_p
        min_p: Minimum probability threshold relative to the max probability

    Returns:
        tuple of (sampled_token, probabilities) where:
        - sampled_token: shape [batch_size, 1]
        - probabilities: shape [batch_size, vocab_size]
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Temperature = 0 means greedy
        return logits.argmax(dim=-1, keepdim=True), F.softmax(logits, dim=-1)

    # Apply min-p filtering
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        min_p_threshold = max_probs * min_p
        logits = torch.where(
            probs >= min_p_threshold,
            logits,
            torch.tensor(float("-inf")).to(logits.device),
        )

    # Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, top_k_indices, top_k_logits)

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1)

    return sampled_token, probs


def compute_logu(
    logits: torch.Tensor, topk: int = 10
) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    Calculate log-u score of the prediction distribution.

    This computes aleatoric and epistemic uncertainty based on the
    logits distribution using the log-u method.

    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider

    Returns:
        Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        Each is a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)

    # Get top-k logits and their indices
    topk_logits, _ = torch.topk(logits, topk, dim=-1)  # [batch_size, topk]

    # Calculate sum of logits (S)
    alpha = torch.sum(topk_logits, dim=-1, keepdim=True)  # [batch_size, 1]

    # Calculate normalized probabilities (p_i = x_i/S)
    probs = topk_logits / alpha  # [batch_size, topk]

    # Calculate digamma terms
    digamma_xi = torch.digamma(topk_logits + 1)  # ψ(x_i + 1)
    digamma_sum = torch.digamma(alpha + 1)  # ψ(S + 1)

    # Calculate aleatoric uncertainty efficiently
    # AU = -∑(p_i * (ψ(x_i + 1) - ψ(S + 1)))
    aleatoric_uncertainty = -torch.sum(
        probs * (digamma_xi - digamma_sum), dim=-1
    )  # [batch_size]

    # Calculate epistemic uncertainty
    # EU = K / (S + K)
    epistemic_uncertainty = topk / (alpha.squeeze(-1) + topk)  # [batch_size]

    if is_single_input:
        return aleatoric_uncertainty.item(), epistemic_uncertainty.item()
    else:
        return aleatoric_uncertainty, epistemic_uncertainty


def calculate_token_entropy(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Calculate token entropy from logits.

    Args:
        logits: Tensor of shape (vocab_size,) or (batch_size, vocab_size)
        temperature: Decoding temperature, default 1.0

    Returns:
        entropy: Scalar or Tensor of entropy values
    """
    # Handle temperature = 0 case (greedy decoding has zero entropy)
    if temperature == 0.0:
        if logits.dim() == 1:
            return torch.tensor(0.0, device=logits.device)
        else:
            return torch.zeros(logits.shape[0], device=logits.device)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Calculate probability distribution (Softmax)
    probs = F.softmax(scaled_logits, dim=-1)

    # Calculate entropy H = -sum(p * log(p))
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy
