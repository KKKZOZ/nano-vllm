import atexit
import time
from dataclasses import fields

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from hybrid_generator.backends.base import ModelBackend
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.logger import logger


class NanovLLMBackend(ModelBackend):
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # logger.info(f"LLMEngine config: {config_kwargs}")
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # self.scheduler = Scheduler(config)
        atexit.register(self.exit)

        # add for backend
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.eos = config.eos
        self.seq_map = {}

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def forward(
        self,
        seq_id: int,
        token_ids: list[int],
    ) -> torch.Tensor:
        is_prefill = False
        is_extend = False

        if seq_id not in self.seq_map:
            seq = Sequence(seq_id, token_ids, SamplingParams())
            self.seq_map[seq.seq_id] = seq
            self.block_manager.allocate(seq)
            is_prefill = True
        else:
            seq = self.seq_map[seq_id]

            if len(token_ids) == 1:
                seq.append_token(token_ids[0])
                self.block_manager.may_append(seq)
                is_prefill = False

            else:
                old_num_tokens = seq.num_tokens

                for token_id in token_ids:
                    seq.append_token(token_id)
                    # may_append can only handle one token at a time
                    self.block_manager.may_append(seq)
                seq.num_cached_tokens = old_num_tokens
                is_extend = True

        logits = self.model_runner.call("run", [seq], is_prefill, is_extend, False)
        return logits

    def free(self, seq_id: int):
        seq = self.seq_map.pop(seq_id)
        self.block_manager.deallocate(seq)

    def rollback(self, seq_id: int, target_len: int):
        pass

    def generate_v0(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> dict[str, str | list[int]]:
        prompt = self.tokenizer.encode(prompt)
        seq = Sequence(None, prompt, sampling_params)

        is_prefill = True
        start_time = time.perf_counter()
        # allocate blocks
        self.block_manager.allocate(seq)
        while not seq.is_finished:
            # only one seq here

            [token_id] = self.model_runner.call("run", [seq], is_prefill)
            if is_prefill:
                prefill_time = time.perf_counter() - start_time
            is_prefill = False

            seq.append_token(token_id)
            self.block_manager.may_append(seq)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                # self.running.remove(seq)

        output = {
            "text": self.tokenizer.decode(seq.completion_token_ids),
            "token_ids": seq.completion_token_ids,
        }
        decode_time = time.perf_counter() - start_time - prefill_time
        logger.info(f"Prefill speed: {len(prompt) / prefill_time:.2f} tokens/s")
        logger.info(
            f"Decode speed: {seq.num_completion_tokens / decode_time:.2f} tokens/s"
        )
        return output
