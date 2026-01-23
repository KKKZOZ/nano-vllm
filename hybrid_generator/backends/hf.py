from typing import Dict

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache  # Requires transformers >= 4.36

from hybrid_generator.backends.base import ModelBackend


class HFBackend(ModelBackend):
    def __init__(
        self,
        model_path: str,
        device: torch.device | str | int | None = None,
        dtype=torch.bfloat16,
    ):
        super().__init__(device)
        device_str = str(self.device)
        print(f"Loading model {model_path} to {device_str}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": device_str},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        # Map seq_id to DynamicCache objects for persistent KV state management
        self.cache_map: Dict[int, DynamicCache] = {}

    def forward(self, seq_id: int, token_ids: list[int]) -> torch.Tensor:
        # Initialize DynamicCache for the prefill phase
        if seq_id not in self.cache_map:
            self.cache_map[seq_id] = DynamicCache()

        cache = self.cache_map[seq_id]
        input_tensor = torch.tensor([token_ids], device=self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_tensor, past_key_values=cache, use_cache=True
            )

        # DynamicCache is updated in-place; no manual past_key_values assignment needed
        return outputs.logits[0]

    def rollback(self, seq_id: int, target_len: int):
        """
        Truncates the KV cache to a target sequence length.
        """
        if seq_id in self.cache_map:
            cache = self.cache_map[seq_id]
            cache.crop(target_len)

    def free(self, seq_id: int):
        if seq_id in self.cache_map:
            del self.cache_map[seq_id]

    def report_stats(self) -> dict:
        return {}

    def exit(self):
        """Release all resources and clear GPU memory."""
        # Clear all caches
        self.cache_map.clear()

        # Delete model
        if hasattr(self, "model"):
            del self.model

        # Clear CUDA cache if using CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Synchronize to ensure cleanup is complete
            torch.cuda.synchronize(self.device)
