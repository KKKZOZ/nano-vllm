import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.logger import logger


class LLMEngine:
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
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(None, prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        logger.debug(f"step: seqs: {seqs}, token_ids: {token_ids}")
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        decode_tokens_total = 0
        decode_start_time = None
        decode_end_time = None
        i = 0
        while not self.is_finished():
            i += 1
            logger.debug(f"Step {i}:")
            t = perf_counter()
            output, num_tokens = self.step()
            step_time = perf_counter() - t
            logger.debug(f"output: {output}, num_tokens: {num_tokens}")
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / max(step_time, 1e-9)
                else:
                    decode_throughput = -num_tokens / max(step_time, 1e-9)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            if num_tokens < 0:
                if decode_start_time is None:
                    decode_start_time = t
                decode_tokens_total += -num_tokens
            for seq_id, token_ids in output:
                logger.debug(f"Step {i}: seq_id: {seq_id}, token_ids: {token_ids}")
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if decode_start_time is not None:
            decode_end_time = perf_counter()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if decode_start_time is not None and decode_end_time is not None:
            decode_time_total = max(decode_end_time - decode_start_time, 1e-9)
            decode_speed = decode_tokens_total / decode_time_total
            logger.info(f"Decode speed: {decode_speed:.2f} tok/s")
        if use_tqdm:
            pbar.close()
        return outputs
