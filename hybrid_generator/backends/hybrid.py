from enum import Enum

import torch

from hybrid_generator.backends.async_wrapper import AsyncBackendWrapper
from hybrid_generator.backends.base import ModelBackend


class BackendId(Enum):
    SLM = "slm"
    LLM = "llm"


class HybridBackend:
    """
    Combine two existing ModelBackend instances (SLM + LLM) into one interface.
    """

    def __init__(self, slm_backend: ModelBackend, llm_backend: ModelBackend):
        self.slm_backend = slm_backend
        self.llm_backend = llm_backend

    def _get_backend(self, backend_id: BackendId) -> ModelBackend:
        if backend_id == BackendId.SLM:
            return self.slm_backend
        if backend_id == BackendId.LLM:
            return self.llm_backend
        raise ValueError(f"Unknown backend_id: {backend_id}")

    def forward(
        self,
        backend_id: BackendId,
        seq_id: int,
        token_ids: list[int],
    ) -> torch.Tensor:
        return self._get_backend(backend_id).forward(seq_id, token_ids)

    def rollback(self, backend_id: BackendId, seq_id: int, target_len: int):
        self._get_backend(backend_id).rollback(seq_id, target_len)

    def free(self, backend_id: BackendId, seq_id: int):
        self._get_backend(backend_id).free(seq_id)

    def report_stats(self) -> dict:
        return {
            BackendId.SLM.value: self.slm_backend.report_stats(),
            BackendId.LLM.value: self.llm_backend.report_stats(),
        }

    def print_stats(self):
        if hasattr(self.slm_backend, "print_stats"):
            self.slm_backend.print_stats()
        if hasattr(self.llm_backend, "print_stats"):
            self.llm_backend.print_stats()

    def exit(self):
        if hasattr(self.slm_backend, "exit"):
            self.slm_backend.exit()
        if hasattr(self.llm_backend, "exit"):
            self.llm_backend.exit()


class AsyncHybridBackend(HybridBackend):
    """
    HybridBackend with async support for concurrent execution.

    Wraps both SLM and LLM backends with AsyncBackendWrapper to enable
    concurrent forward calls using asyncio.
    """

    def __init__(self, slm_backend: ModelBackend, llm_backend: ModelBackend):
        """
        Initialize async hybrid backend.

        Args:
            slm_backend: Small language model backend
            llm_backend: Large language model backend
        """
        super().__init__(slm_backend, llm_backend)

        # Wrap backends for async execution
        self.slm_async = AsyncBackendWrapper(slm_backend, max_workers=1)
        self.llm_async = AsyncBackendWrapper(llm_backend, max_workers=1)

    async def forward_async(
        self, backend_id: BackendId, seq_id: int, token_ids: list[int]
    ) -> torch.Tensor:
        """
        Async forward for concurrent execution.

        Args:
            backend_id: Which backend to use (SLM or LLM)
            seq_id: Sequence ID
            token_ids: Token IDs to process

        Returns:
            Logits tensor of shape [len(token_ids), vocab_size]
        """
        if backend_id == BackendId.SLM:
            return await self.slm_async.forward_async(seq_id, token_ids)
        else:
            return await self.llm_async.forward_async(seq_id, token_ids)

    def shutdown(self):
        """Shutdown async wrappers."""
        self.slm_async.shutdown()
        self.llm_async.shutdown()

    def exit(self):
        """Shutdown async wrappers and exit underlying backends."""
        # First shutdown the async wrappers (thread pools)
        self.shutdown()
        # Then exit the underlying backends
        super().exit()
