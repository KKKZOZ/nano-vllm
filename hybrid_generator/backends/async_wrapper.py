"""
Async wrapper for synchronous backends.

Provides async interface for backends that only support synchronous operations
using ThreadPoolExecutor.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch

from hybrid_generator.backends.base import ModelBackend


class AsyncBackendWrapper:
    """
    Wraps a synchronous ModelBackend to provide async interface.

    Uses ThreadPoolExecutor to run synchronous forward() calls in background threads,
    enabling concurrent execution with other async operations.
    """

    def __init__(self, backend: ModelBackend, max_workers: int = 1):
        """
        Initialize async wrapper.

        Args:
            backend: The synchronous backend to wrap
            max_workers: Number of worker threads (default: 1)
        """
        self.backend = backend
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def forward_async(self, seq_id: int, token_ids: list[int]) -> torch.Tensor:
        """
        Async wrapper for forward() using ThreadPoolExecutor.

        Args:
            seq_id: Sequence ID for KV cache management
            token_ids: List of token IDs to process

        Returns:
            Logits tensor of shape [len(token_ids), vocab_size]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._forward_sync_wrapper, seq_id, token_ids
        )

    def _forward_sync_wrapper(self, seq_id: int, token_ids: list[int]) -> torch.Tensor:
        """
        Wrapper for thread-safe forward call.

        Sets CUDA device context if backend has device attribute.
        """
        # Set CUDA device context for thread safety
        if hasattr(self.backend, "device"):
            device = self.backend.device
            # Handle both torch.device objects and string device specifications
            if isinstance(device, torch.device):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
            elif isinstance(device, str) and device.startswith("cuda"):
                torch.cuda.set_device(device)

        return self.backend.forward(seq_id, token_ids)

    def forward_sync(self, seq_id: int, token_ids: list[int]) -> torch.Tensor:
        """
        Synchronous forward for compatibility.

        Args:
            seq_id: Sequence ID
            token_ids: Token IDs to process

        Returns:
            Logits tensor
        """
        return self.backend.forward(seq_id, token_ids)

    def rollback(self, seq_id: int, target_len: int):
        """Rollback KV cache (synchronous operation)."""
        self.backend.rollback(seq_id, target_len)

    def free(self, seq_id: int):
        """Free KV cache (synchronous operation)."""
        self.backend.free(seq_id)

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
