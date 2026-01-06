import torch

from nanovllm.backend import Backend


class TwinBackend:
    """
    Manage two Backend instances (e.g., SLM + LLM) under a single interface.

    Notes:
    - This is designed for coordinating two models in one process.
    - tensor_parallel_size > 1 is not supported yet because ModelRunner relies
      on a single global process group and shared memory name.
    """

    def __init__(
        self,
        slm_model: str,
        llm_model: str,
        *,
        slm_kwargs: dict | None = None,
        llm_kwargs: dict | None = None,
        **shared_kwargs,
    ):
        if slm_model is None or llm_model is None:
            raise ValueError("TwinBackend requires both slm_model and llm_model.")

        slm_kwargs = slm_kwargs or {}
        llm_kwargs = llm_kwargs or {}
        slm_args = {**shared_kwargs, **slm_kwargs}
        llm_args = {**shared_kwargs, **llm_kwargs}

        slm_tp = slm_args.get("tensor_parallel_size", 1)
        llm_tp = llm_args.get("tensor_parallel_size", 1)
        if slm_tp > 1 or llm_tp > 1:
            raise ValueError(
                "TwinBackend currently supports tensor_parallel_size=1 only."
            )

        self.slm = Backend(slm_model, **slm_args)
        self.llm = Backend(llm_model, **llm_args)

    def _get_backend(self, which: str) -> Backend:
        if which == "slm":
            return self.slm
        if which == "llm":
            return self.llm
        raise ValueError(f"Unknown backend key: {which}. Use 'slm' or 'llm'.")

    def forward(
        self,
        which: str,
        seq_id: int,
        token_ids: list[int],
    ) -> torch.Tensor:
        return self._get_backend(which).forward(seq_id, token_ids)

    def free(self, which: str, seq_id: int):
        self._get_backend(which).free(seq_id)

    def rollback(self, which: str, seq_id: int, target_len: int):
        self._get_backend(which).rollback(seq_id, target_len)

    def report_stats(self) -> dict:
        return {
            "slm": self.slm.report_stats(),
            "llm": self.llm.report_stats(),
        }

    def print_stats(self):
        print("\n" + "=" * 80)
        print("TwinBackend Statistics")
        print("=" * 80)
        print("\n[SLM]")
        self.slm.print_stats()
        print("\n[LLM]")
        self.llm.print_stats()

    def exit(self):
        self.slm.exit()
        self.llm.exit()
