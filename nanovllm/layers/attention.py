import torch
from torch import nn

from nanovllm.layers.attn_backend import FlashAttnBackend, NaiveAttnBackend


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.backend = FlashAttnBackend(num_heads, head_dim, scale, num_kv_heads)

    @property
    def k_cache(self):
        return self.backend.k_cache

    @k_cache.setter
    def k_cache(self, value):
        self.backend.k_cache = value

    @property
    def v_cache(self):
        return self.backend.v_cache

    @v_cache.setter
    def v_cache(self, value):
        self.backend.v_cache = value

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.backend.forward(q, k, v)
