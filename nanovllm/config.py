import os
from dataclasses import dataclass
from transformers import AutoConfig

USE_FLASH_INFER = True


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    block_size: int = 16
    num_kvcache_blocks: int = -1
    decode_window_blocks: int | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
        self.decode_window_blocks = 4
        if self.decode_window_blocks is not None:
            assert self.decode_window_blocks > 0
            max_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
            assert self.decode_window_blocks <= max_blocks


_global_config: Config | None = None


def init_config(model: str, **kwargs) -> Config:
    global _global_config
    _global_config = Config(model=model, **kwargs)
    return _global_config


def get_config() -> Config:
    if _global_config is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _global_config
