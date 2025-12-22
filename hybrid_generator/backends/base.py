from abc import ABC, abstractmethod

import torch


class ModelBackend(ABC):
    """
    Abstract base class for model inference backends.
    """

    @abstractmethod
    def forward(self, seq_id: int, token_ids: list[int]) -> torch.Tensor:
        """
        Core forward pass unifying Prefill, Extend, Decode, and Verify phases.

        This method processes the input `token_ids` by appending them to the KV cache
        associated with `seq_id` and executing a forward pass.

        **Behavior:**
        The method always returns the logits corresponding to the input `token_ids`.
        - If `token_ids` has length L, the output tensor will have shape [L, vocab_size].
        - The i-th logit corresponds to the prediction for the token *after* token_ids[i].

        **Usage Patterns:**

        1. **Prefill**:
           - Input: Complete prompt `[p1, p2, ..., pn]`
           - Output: Logits `[l1, l2, ..., ln]` (Shape: [n, vocab])

        2. **Extend (Catch-up) / Verify**:
           - Input: New tokens `[t1, t2, t3]`
           - Output: Logits `[o1, o2, o3]` (Shape: [3, vocab])
           - Used for synchronizing state or verifying draft tokens.

        3. **Decode**:
           - Input: Single token `[t]`
           - Output: Logit `[o]` (Shape: [1, vocab])

        Args:
            seq_id (int): The unique identifier for the sequence/session.
            token_ids (list[int]): A list of new token IDs to append.

        Returns:
            torch.Tensor: A tensor of shape `[len(token_ids), vocab_size]`.
        """
        pass

    @abstractmethod
    def rollback(self, seq_id: int, target_len: int):
        """
        Truncates the KV cache for a specific request to a target length.
        Required for Speculative Decoding rejection sampling.
        """
        pass

    @abstractmethod
    def free(self, seq_id: int):
        """Releases resources associated with the request ID."""
        pass
