import torch
import triton
import triton.language as tl

from nanovllm.layers.attn_backend.base import AttnBackend
from nanovllm.utils.context import get_context


@triton.jit
def gather_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    out_k_ptr,
    out_v_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    cache_offsets = slot * D + offsets
    out_offsets = idx * D + offsets
    tl.store(out_k_ptr + out_offsets, tl.load(k_cache_ptr + cache_offsets))
    tl.store(out_v_ptr + out_offsets, tl.load(v_cache_ptr + cache_offsets))


class NaiveAttnBackend(AttnBackend):
    """朴素 PyTorch attention 后端，使用 Triton 收集 paged KV。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scratch_k: torch.Tensor | None = None
        self.scratch_v: torch.Tensor | None = None

    def _ensure_scratch(
        self, num_tokens: int, device: torch.device, dtype: torch.dtype
    ):
        if (
            self.scratch_k is None
            or self.scratch_k.numel() < num_tokens * self.num_kv_heads * self.head_dim
            or self.scratch_k.device != device
            or self.scratch_k.dtype != dtype
        ):
            self.scratch_k = torch.empty(
                num_tokens,
                self.num_kv_heads * self.head_dim,
                device=device,
                dtype=dtype,
            )
            self.scratch_v = torch.empty_like(self.scratch_k)

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
        flat_k = k.reshape(k.size(0), self.num_kv_heads * self.head_dim)
        flat_v = v.reshape(v.size(0), self.num_kv_heads * self.head_dim)
        cache_k = self.k_cache.view(-1, self.num_kv_heads * self.head_dim)
        cache_v = self.v_cache.view(-1, self.num_kv_heads * self.head_dim)
        indices = slot_mapping.long()
        cache_k[indices] = flat_k
        cache_v[indices] = flat_v

    def _gather_kv(
        self, slot_mapping: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = slot_mapping.numel()
        if num_tokens == 0:
            empty = torch.empty(
                0, self.num_kv_heads, self.head_dim, device=device, dtype=dtype
            )
            return empty, empty
        self._ensure_scratch(num_tokens, device, dtype)
        k_cache_flat = self.k_cache.reshape(-1, self.num_kv_heads * self.head_dim)
        v_cache_flat = self.v_cache.reshape(-1, self.num_kv_heads * self.head_dim)
        gather_kvcache_kernel[(num_tokens,)](
            k_cache_flat,
            v_cache_flat,
            self.scratch_k.reshape(-1),
            self.scratch_v.reshape(-1),
            slot_mapping,
            self.num_kv_heads * self.head_dim,
        )
        k_out = self.scratch_k[:num_tokens].view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_out = self.scratch_v[:num_tokens].view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        return k_out, v_out

    def _repeat_kv_for_gqa(self, kv: torch.Tensor) -> torch.Tensor:
        if self.num_heads == self.num_kv_heads:
            return kv
        group_size = self.num_heads // self.num_kv_heads
        return kv.repeat_interleave(group_size, dim=1)

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> torch.Tensor:
        # q: [Lq, num_heads, head_dim], k/v: [Lk, num_kv_heads, head_dim]
        k_exp = self._repeat_kv_for_gqa(k)
        v_exp = self._repeat_kv_for_gqa(v)
        q_pos = start + torch.arange(q.size(0), device=q.device)
        k_pos = torch.arange(k.size(0), device=k.device)
        causal = q_pos[:, None] >= k_pos[None, :]
        scores = torch.einsum("qhd,khd->qhk", q, k_exp) * self.scale
        scores = scores.masked_fill(~causal.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("qhk,khd->qhd", attn, v_exp)

    def _build_slot_mapping_from_blocks(
        self, block_tables: torch.Tensor, seqlens: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        mappings = []
        for seq_idx in range(block_tables.size(0)):
            seqlen = int(seqlens[seq_idx].item())
            if seqlen == 0:
                continue
            positions = torch.arange(
                seqlen, device=block_tables.device, dtype=torch.int32
            )
            block_ids = block_tables[seq_idx, positions // block_size]
            slots = block_ids * block_size + (positions % block_size)
            mappings.append(slots)
        if not mappings:
            return torch.empty(0, device=block_tables.device, dtype=torch.int32)
        return torch.cat(mappings, dim=0)

    def _block_owner(self, block_tables: torch.Tensor) -> dict[int, int]:
        owner: dict[int, int] = {}
        for seq_idx in range(block_tables.size(0)):
            for block in block_tables[seq_idx].tolist():
                if block == -1:
                    continue
                owner[int(block)] = seq_idx
        return owner

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()
        device = q.device
        dtype = q.dtype

        # Write new KV into cache if available (uses scratch buffers)
        if context.slot_mapping is not None:
            num_slots = context.slot_mapping.numel()
            if num_slots:
                self._ensure_scratch(num_slots, device, dtype)
                self.store_kv_cache(k, v, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is None:
                # No prefix cache; use provided k/v directly
                total_k = k
                total_v = v
            else:
                block_size = self.k_cache.size(1)
                slot_mapping = self._build_slot_mapping_from_blocks(
                    context.block_tables,
                    context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1],
                    block_size,
                )
                total_k, total_v = self._gather_kv(slot_mapping, device, dtype)
            outputs = []
            cu_q = context.cu_seqlens_q
            cu_k = context.cu_seqlens_k
            for i in range(cu_q.numel() - 1):
                q_slice = q[cu_q[i] : cu_q[i + 1]]
                k_slice = total_k[cu_k[i] : cu_k[i + 1]]
                v_slice = total_v[cu_k[i] : cu_k[i + 1]]
                start = k_slice.size(0) - q_slice.size(0)
                outputs.append(self._attend(q_slice, k_slice, v_slice, start))
            return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

        # Decode or extend
        block_size = self.k_cache.size(1)
        block_tables = context.block_tables
        seqlens = context.context_lens
        slot_mapping = self._build_slot_mapping_from_blocks(
            block_tables, seqlens, block_size
        )
        gathered_k, gathered_v = self._gather_kv(slot_mapping, device, dtype)

        if context.is_extend:
            # Map tokens back to sequences using block ownership
            owner = self._block_owner(block_tables)
            outputs = torch.empty_like(q)
            q_indices_per_seq: dict[int, list[int]] = {}
            for idx, slot in enumerate(context.slot_mapping.tolist()):
                seq_id = owner.get(int(slot // block_size))
                q_indices_per_seq.setdefault(seq_id, []).append(idx)
            cursor = 0
            for seq_idx, seqlen in enumerate(seqlens.tolist()):
                k_slice = gathered_k[cursor : cursor + seqlen]
                v_slice = gathered_v[cursor : cursor + seqlen]
                cursor += seqlen
                token_indices = q_indices_per_seq.get(seq_idx, [])
                if not token_indices:
                    continue
                q_slice = q[token_indices]
                start = seqlen - len(token_indices)
                out_slice = self._attend(q_slice, k_slice, v_slice, start)
                outputs[token_indices] = out_slice
            return outputs

        # Decode: one token per sequence
        outputs = torch.empty_like(q)
        cursor = 0
        for seq_idx, seqlen in enumerate(seqlens.tolist()):
            k_slice = gathered_k[cursor : cursor + seqlen]
            v_slice = gathered_v[cursor : cursor + seqlen]
            cursor += seqlen
            q_slice = q[seq_idx : seq_idx + 1]
            start = seqlen - 1
            out_slice = self._attend(q_slice, k_slice, v_slice, start)
            outputs[seq_idx : seq_idx + 1] = out_slice
        return outputs
