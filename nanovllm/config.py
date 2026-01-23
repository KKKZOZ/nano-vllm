import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.utils.logger import logger


@dataclass
class Config:
    model: str
    device: int | None = None
    max_num_batched_tokens: int = 40960
    max_num_seqs: int = 512
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.4
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_extend_cudagraph: bool = True
    max_extend_len: int = 20
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        logger.info("GPU Memory Utilization: %.2f", self.gpu_memory_utilization)
        assert os.path.isdir(self.model)
        assert self.device is None or self.device >= 0
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
