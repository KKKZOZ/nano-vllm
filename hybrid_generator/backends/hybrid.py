from enum import Enum

import torch

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
