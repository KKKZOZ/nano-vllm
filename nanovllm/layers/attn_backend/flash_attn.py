import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.attn_backend.base import AttnBackend
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,  # ty:ignore[invalid-argument-type]
    )


class FlashAttnBackend(AttnBackend):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()
        self.store_kv_cache(k, v, context.slot_mapping)
        if context.is_prefill and not context.is_extend:
            return self._prefill_forward(q, k, v, context)
        return self._decode_forward(q, context)

    def store_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor | None
    ) -> None:
        if (
            self.k_cache.numel() == 0
            or self.v_cache.numel() == 0
            or slot_mapping is None
            or slot_mapping.numel() == 0
            or k.numel() == 0
            or v.numel() == 0
        ):
            return
        store_kvcache(k, v, self.k_cache, self.v_cache, slot_mapping)

    def _prefill_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context
    ) -> torch.Tensor:
        if context.block_tables is not None:
            k = self.k_cache
            v = self.v_cache
        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=context.block_tables,
        )

    def _decode_forward(self, q: torch.Tensor, context) -> torch.Tensor:
        if context.is_extend:
            q = q.unsqueeze(0)
        else:
            q = q.unsqueeze(1)

        output = flash_attn_with_kvcache(
            q,
            self.k_cache,
            self.v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )

        if context.is_extend:
            return output.squeeze(0)
        return output.squeeze(1)
