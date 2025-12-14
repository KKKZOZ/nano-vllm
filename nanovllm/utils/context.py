from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # A block is composed of *block_size* consecutive slots
    # Mapping token to slots
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    # Mapping sequence to blocks
    # block_tables = torch.tensor([
    #     [1, 3, 5, -1],  # 序列 0: 使用物理块 1, 3, 5
    #     [2, 4, -1, -1], # 序列 1: 使用物理块 2, 4
    #     [0, 6, 7, 8],   # 序列 2: 使用物理块 0, 6, 7, 8
    # ])
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

"""

Decoding
block_tables = torch.tensor([
    [1, 3, 5, -1],  # 序列 0
    [2, 4, -1, -1], # 序列 1
    [0, 6, 7, 8],   # 序列 2
])

# 序列 0: 长度 9 → 第 9 个位置在逻辑块 2（第 3 个块）的位置 1
#         逻辑块 2 对应物理块 5
slot_mapping[0] = block_tables[0][2] * 4 + 1 = 5 * 4 + 1 = 21

# 序列 1: 长度 6 → 第 6 个位置在逻辑块 1（第 2 个块）的位置 2
#         逻辑块 1 对应物理块 4
slot_mapping[1] = block_tables[1][1] * 4 + 2 = 4 * 4 + 2 = 18

# 序列 2: 长度 15 → 第 15 个位置在逻辑块 3（第 4 个块）的位置 3
#         逻辑块 3 对应物理块 8
slot_mapping[2] = block_tables[2][3] * 4 + 3 = 8 * 4 + 3 = 35

slot_mapping = torch.tensor([21, 18, 35])

Prefilling
block_tables[0] = [1, 3, 5, -1]

slot_mapping = torch.tensor([
    # 逻辑块 0 → 物理块 1
    1*4 + 0,  # token_0 → 物理位置 4
    1*4 + 1,  # token_1 → 物理位置 5
    1*4 + 2,  # token_2 → 物理位置 6
    1*4 + 3,  # token_3 → 物理位置 7
    # 逻辑块 1 → 物理块 3
    3*4 + 0,  # token_4 → 物理位置 12
    3*4 + 1,  # token_5 → 物理位置 13
    3*4 + 2,  # token_6 → 物理位置 14
    3*4 + 3,  # token_7 → 物理位置 15
    # 逻辑块 2 → 物理块 5
    5*4 + 0,  # token_8 → 物理位置 20
])
# = [4, 5, 6, 7, 12, 13, 14, 15, 20]

"""
