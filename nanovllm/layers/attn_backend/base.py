from abc import ABC, abstractmethod

import torch


class AttnBackend(ABC):
    """Attention 后端抽象基类。

    所有后端必须实现:
    - store_kv_cache(): 将 K, V 存储到 KV cache
    - forward(): 计算 attention 输出

    Tensor Shapes:
    -------------
    输入:
        q: [num_tokens, num_heads, head_dim]
            - num_tokens: batch 中所有 token 的总数（packed format）
            - num_heads: 注意力头数量（已考虑 tensor parallelism）
            - head_dim: 每个头的维度

        k: [num_tokens, num_kv_heads, head_dim]
            - num_kv_heads: KV 头数量（GQA 时 num_kv_heads < num_heads）

        v: [num_tokens, num_kv_heads, head_dim]
            - 与 k 相同的 shape

    输出:
        output: [num_tokens, num_heads, head_dim]
            - 与 q 相同的 shape

    Context (通过 get_context() 获取):
        - is_prefill: bool - 是否为 prefill 阶段
        - is_extend: bool - 是否为 extend 模式
        - cu_seqlens_q: [batch_size + 1] - query 累积序列长度
        - cu_seqlens_k: [batch_size + 1] - key 累积序列长度
        - max_seqlen_q: int - 最大 query 序列长度
        - max_seqlen_k: int - 最大 key 序列长度
        - slot_mapping: [num_tokens] - token 到 cache slot 的映射
        - context_lens: [batch_size] - 每个序列的缓存长度
        - block_tables: [batch_size, max_blocks] - 逻辑块到物理块的映射
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """计算 attention。

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, num_kv_heads, head_dim]
            v: Value tensor [num_tokens, num_kv_heads, head_dim]

        Returns:
            output: Attention 输出 [num_tokens, num_heads, head_dim]
        """
        ...

    @abstractmethod
    def store_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor | None
    ):
        """将 K, V 写入 KV cache。"""
        ...

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """设置 KV cache 引用。

        Args:
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        """
        self.k_cache = k_cache
        self.v_cache = v_cache
