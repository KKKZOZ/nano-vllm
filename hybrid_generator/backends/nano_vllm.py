
import torch

from hybrid_generator.backends.base import ModelBackend
from nanovllm.backend import Backend
from nanovllm.sampling_params import SamplingParams


class NanovLLMBackend(ModelBackend):
    def __init__(self, model: str | None = None, *, backend: Backend | None = None, **kwargs):
        if backend is None:
            if model is None:
                raise ValueError("NanovLLMBackend requires a model or a backend.")
            self.backend = Backend(model, **kwargs)
        else:
            self.backend = backend

    def exit(self):
        self.backend.exit()

    def forward(
        self,
        seq_id: int,
        token_ids: list[int],
    ) -> torch.Tensor:
        return self.backend.forward(seq_id, token_ids)

    def free(self, seq_id: int):
        self.backend.free(seq_id)

    def rollback(self, seq_id: int, target_len: int):
        pass

    def report_stats(self) -> dict:
        return self.backend.report_stats()

    def print_stats(self):
        self.backend.print_stats()

    def generate_v0(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> dict[str, str | list[int]]:
        return self.backend.generate_v0(prompt, sampling_params)
