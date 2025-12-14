import flashinfer
import torch
from torch import device, nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.config import Config, get_config


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
    """
    key_ptr & value_ptr: [N, D]
    k_cache_ptr & v_cache_ptr: [num_slots, D]
    slot_mapping_ptr: [N], each element is in [-1, num_slots-1]
    we need to store key[i], value[i] to k_cache[slot_mapping_ptr[i]], v_cache[slot_mapping_ptr[i]]
    """
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
    """
    key & value:  [N, num_heads, head_dim]
    kv_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping can indicate a (num_blocks, block_size) tensor
    we do not care the (num_kv_headsm, head_dim) parts, since they are contiguous in memory
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


def store_kvcache_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Store key/value into KV cache based on slot_mapping.

    Args:
        key: (N, num_kv_heads, head_dim)
        value: (N, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (N,) - logical slot index, -1 means skip
    """
    # Filter out invalid slots
    valid_mask = slot_mapping != -1
    valid_slots = slot_mapping[valid_mask]
    valid_keys = key[valid_mask]
    valid_values = value[valid_mask]

    if valid_slots.numel() == 0:
        return

    # Convert slot index to (block_idx, offset_in_block)
    block_size = k_cache.shape[1]
    block_idx = valid_slots // block_size
    offset_in_block = valid_slots % block_size

    # Store into cache
    k_cache[block_idx, offset_in_block] = valid_keys
    v_cache[block_idx, offset_in_block] = valid_values


_GLOBAL_WRAPPERS = {
    "prefill_paged": None,
    "prefill_ragged": None,
    "decode": None,
    "workspace_buffers": None,
}
# Share decode plan across layers to avoid redundant plan() in every forward
_GLOBAL_PLAN_CACHE = {"decode": None}


def get_flashinfer_wrappers(device="cuda"):
    """Get global shared flashinfer wrappers (singleton)"""
    global _GLOBAL_WRAPPERS

    if _GLOBAL_WRAPPERS["workspace_buffers"] is None:
        # Initialize once for all layers
        workspace_size = 128 * 1024 * 1024  # 128MB

        _GLOBAL_WRAPPERS["workspace_buffers"] = {
            "prefill_paged": torch.empty(
                workspace_size, dtype=torch.uint8, device=device
            ),
            "prefill_ragged": torch.empty(
                workspace_size, dtype=torch.uint8, device=device
            ),
            "decode": torch.empty(workspace_size, dtype=torch.uint8, device=device),
        }

        _GLOBAL_WRAPPERS["prefill_paged"] = (
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                _GLOBAL_WRAPPERS["workspace_buffers"]["prefill_paged"], "NHD"
            )
        )

        _GLOBAL_WRAPPERS["prefill_ragged"] = (
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                _GLOBAL_WRAPPERS["workspace_buffers"]["prefill_ragged"], "NHD"
            )
        )

        _GLOBAL_WRAPPERS["decode"] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _GLOBAL_WRAPPERS["workspace_buffers"]["decode"], "NHD"
        )

    return (
        _GLOBAL_WRAPPERS["prefill_paged"],
        _GLOBAL_WRAPPERS["prefill_ragged"],
        _GLOBAL_WRAPPERS["decode"],
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        config = get_config()
        self.block_size = config.block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache = None
        self._kv_cache_view = None

        # Cache for plan states to avoid redundant plan() calls
        self._last_prefill_paged_params = None
        self._last_prefill_ragged_params = None
        self._last_decode_params = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        prefill_wrapper_paged, prefill_wrapper_ragged, decode_wrapper = (
            get_flashinfer_wrappers(q.device)
        )

        if context.is_prefill:
            if context.block_tables is not None:
                # Use paged KV cache (prefix cache scenario)
                # Stack k_cache and v_cache for flashinfer
                # kv_cache = torch.stack([k_cache, v_cache], dim=1)
                kv_cache = (k_cache, v_cache)

                kv_indices, kv_indptr, kv_last_page_len = convert_block_table_to_ragged(
                    context.block_tables, context.context_lens, self.block_size
                )

                # Cache plan state - only replan if parameters changed
                current_params = (
                    tuple(context.cu_seqlens_q.tolist()),
                    tuple(kv_indptr.tolist()),
                    tuple(kv_indices.tolist()),
                    tuple(kv_last_page_len.tolist()),
                    q.dtype,
                )
                if self._last_prefill_paged_params != current_params:
                    prefill_wrapper_paged.plan(
                        qo_indptr=context.cu_seqlens_q,
                        kv_indptr=kv_indptr,
                        kv_indices=kv_indices,
                        kv_last_page_len=kv_last_page_len,
                        num_qo_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        page_size=self.block_size,
                        causal=True,
                        data_type=q.dtype,
                    )
                    self._last_prefill_paged_params = current_params

                o = prefill_wrapper_paged.run(q, kv_cache)
            else:
                # No paged cache, use current k, v directly
                # Use ragged wrapper for variable-length sequences
                # Cache plan state - only replan if parameters changed
                current_params = (
                    tuple(context.cu_seqlens_q.tolist()),
                    tuple(context.cu_seqlens_k.tolist()),
                    q.dtype,
                )
                if self._last_prefill_ragged_params != current_params:
                    prefill_wrapper_ragged.plan(
                        qo_indptr=context.cu_seqlens_q,
                        kv_indptr=context.cu_seqlens_k,
                        num_qo_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        causal=True,
                        q_data_type=q.dtype,
                    )
                    self._last_prefill_ragged_params = current_params

                o = prefill_wrapper_ragged.run(q, k, v)
        else:  # decode
            # # Reuse pre-stacked KV cache view to avoid per-step copies
            # if self._kv_cache_view is None:
            #     if self.kv_cache is not None:
            #         self._kv_cache_view = self.kv_cache.permute(1, 0, 2, 3, 4)
            #     else:
            #         self._kv_cache_view = torch.stack([k_cache, v_cache], dim=1)
            # kv_cache = self._kv_cache_view

            kv_cache = (k_cache, v_cache)

            # Decode-side ragged indices are shared across layers; compute once if missing
            kv_indices = context.kv_indices
            kv_indptr = context.kv_indptr
            kv_last_page_len = context.kv_last_page_len
            if kv_indices is None or kv_indptr is None or kv_last_page_len is None:
                kv_indices, kv_indptr, kv_last_page_len = convert_block_table_to_ragged(
                    context.block_tables, context.context_lens, self.block_size
                )

            # Cache plan state - only replan if parameters changed
            current_params = (
                tuple(kv_indptr.tolist()),
                tuple(kv_indices.tolist()),
                tuple(kv_last_page_len.tolist()),
                q.dtype,
            )
            if _GLOBAL_PLAN_CACHE["decode"] != current_params:
                decode_wrapper.plan(
                    indptr=kv_indptr,
                    indices=kv_indices,
                    last_page_len=kv_last_page_len,
                    num_qo_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.block_size,
                    q_data_type=q.dtype,
                )
                _GLOBAL_PLAN_CACHE["decode"] = current_params

            # flashinfer decode expects q: [batch_size, num_heads, head_dim]
            # not [batch_size, 1, num_heads, head_dim]
            o = decode_wrapper.run(q.squeeze(1), kv_cache)
        return o


# def convert_block_table_to_ragged(block_tables, seq_lens, block_size):
#     """
#     Convert block_tables to ragged format for flashinfer.

#     Args:
#         block_tables: 2D tensor of shape [num_seqs, max_num_blocks]
#         context_lens: list of context lengths for each sequence
#         block_size: size of each block

#     Returns:
#         kv_indices: 1D tensor of concatenated block indices
#         kv_indptr: 1D tensor of starting indices for each sequence
#         kv_last_page_len: 1D tensor of last page lengths for each sequence
#     """
#     kv_indices_list = []
#     kv_indptr = [0]
#     kv_last_page_len = []
#     num_seqs, max_num_blocks = block_tables.shape
#     # there is no "-1" in kv_indices
#     for i, block_table in enumerate(block_tables):
#         valid_mask = block_table != -1
#         valid_kv = block_table[valid_mask]
#         kv_indices_list.append(valid_kv.tolist())
#         kv_indptr.append(kv_indptr[-1] + valid_kv.numel())
#         last_len = (seq_lens[i] - 1) % block_size + 1
#         kv_last_page_len.append(last_len)

#     kv_indices = torch.cat(kv_indices_list)
#     kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=block_table.device)
#     kv_last_page_len = torch.tensor(
#         kv_last_page_len, dtype=torch.int32, device=block_table.device
#     )

#     return kv_indices, kv_indptr, kv_last_page_len


def convert_block_table_to_ragged(block_table, seq_lens, block_size):
    """Convert block_table to ragged format using pure tensor operations"""
    if block_table is None or seq_lens is None:
        return None, None, None

    batch_size = block_table.shape[0]
    device = block_table.device

    # Convert seq_lens to CPU for iteration (if not already)
    # During CUDA graph capture, we need to avoid .item() on GPU tensors
    if torch.is_tensor(seq_lens) and seq_lens.is_cuda:
        seq_lens_cpu = seq_lens.cpu()
    else:
        seq_lens_cpu = seq_lens

    kv_indices_list = []
    kv_indptr = [0]
    kv_last_page_len = []

    for i in range(batch_size):
        seq_len = (
            seq_lens_cpu[i].item()
            if torch.is_tensor(seq_lens_cpu[i])
            else seq_lens_cpu[i]
        )

        if seq_len <= 0:
            kv_last_page_len.append(0)
            kv_indptr.append(
                kv_indptr[-1]
            )  # Keep indptr in sync even for empty sequences
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = block_table[i, :num_blocks]
        valid_blocks = blocks[blocks != -1]

        if valid_blocks.numel() > 0:
            kv_indices_list.append(valid_blocks)

        kv_indptr.append(kv_indptr[-1] + valid_blocks.numel())
        last_len = (seq_len - 1) % block_size + 1
        kv_last_page_len.append(last_len)

    kv_indices = (
        torch.cat(kv_indices_list)
        if kv_indices_list
        else torch.tensor([], dtype=torch.int32, device=device)
    )
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(kv_last_page_len, dtype=torch.int32, device=device)

    return kv_indices, kv_indptr, kv_last_page_len
