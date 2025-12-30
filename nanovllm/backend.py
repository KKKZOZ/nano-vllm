import atexit
import time
from dataclasses import fields

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.logger import logger


class Backend:
    def __init__(self, model, **kwargs):
        self.enable_stats_sync = kwargs.pop("enable_stats_sync", False)
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

        # Statistics tracking
        self.stats = {
            "prefill": {"num_tokens": [], "times": []},
            "extend": {"num_tokens": [], "times": []},
            "decode": {"num_tokens": [], "times": []},
        }

    def exit(self):
        if hasattr(self, "model_runner"):
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
        num_tokens = len(token_ids)

        if seq_id not in self.seq_map:
            seq = Sequence(seq_id, token_ids, SamplingParams())
            self.seq_map[seq.seq_id] = seq
            self.block_manager.allocate(seq)
            is_prefill = True
            phase = "prefill"
        else:
            seq = self.seq_map[seq_id]

            if len(token_ids) == 1:
                seq.append_token(token_ids[0])
                self.block_manager.may_append(seq)
                is_prefill = False
                phase = "decode"

            else:
                old_num_tokens = seq.num_tokens

                for token_id in token_ids:
                    seq.append_token(token_id)
                    # may_append can only handle one token at a time
                    self.block_manager.may_append(seq)
                seq.num_cached_tokens = old_num_tokens
                is_extend = True
                phase = "extend"

        # Record start time (synchronize to include GPU work)
        if self.enable_stats_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        logits = self.model_runner.call("run", [seq], is_prefill, is_extend, False)

        # Record end time and statistics
        if self.enable_stats_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        self.stats[phase]["num_tokens"].append(num_tokens)
        self.stats[phase]["times"].append(elapsed_time)

        return logits

    def free(self, seq_id: int):
        seq = self.seq_map.pop(seq_id)
        self.block_manager.deallocate(seq)

    def rollback(self, seq_id: int, target_len: int):
        pass

    def report_stats(self) -> dict:
        """
        Report statistics for forward operations.

        Returns a dictionary with statistics for prefill, extend, and decode operations:
        - count: number of calls
        - num_tokens: statistics about number of tokens processed
        - times: statistics about time spent (in milliseconds)
        """
        report = {}

        def round4(v):
            return float(round(v, 4))

        for phase in ["prefill", "extend", "decode"]:
            num_tokens_list = self.stats[phase]["num_tokens"]
            times_list = self.stats[phase]["times"]

            if not num_tokens_list:
                report[phase] = {
                    "count": 0,
                    "num_tokens": {},
                    "times": {},
                }
                continue

            num_tokens_array = np.array(num_tokens_list)
            times_array = np.array(times_list)
            times_array_ms = times_array * 1000.0

            report[phase] = {
                "count": len(num_tokens_list),
                "num_tokens": {
                    "mean": round4(np.mean(num_tokens_array)),
                    "median": round4(np.median(num_tokens_array)),
                    "p80": round4(np.percentile(num_tokens_array, 80)),
                    "min": int(np.min(num_tokens_array)),
                    "max": int(np.max(num_tokens_array)),
                },
                "times": {
                    "mean": round4(np.mean(times_array_ms)),
                    "median": round4(np.median(times_array_ms)),
                    "p80": round4(np.percentile(times_array_ms, 80)),
                    "min": round4(np.min(times_array_ms)),
                    "max": round4(np.max(times_array_ms)),
                    "total": round4(np.sum(times_array_ms)),
                },
            }

            # Add throughput for convenience
            if report[phase]["times"]["mean"] > 0:
                report[phase]["throughput"] = {
                    "mean_tokens_per_sec": round4(
                        report[phase]["num_tokens"]["mean"]
                        / (report[phase]["times"]["mean"] / 1000.0)
                    ),
                }

        return report

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.report_stats()

        print("\n" + "=" * 80)
        print("NanovLLM Backend Statistics")
        print("=" * 80)

        for phase in ["prefill", "extend", "decode"]:
            phase_stats = stats[phase]
            print(f"\n{phase.upper()}:")
            print(f"  Total calls: {phase_stats['count']}")

            if phase_stats["count"] > 0:
                print("  Number of tokens:")
                print(
                    f"    Mean: {phase_stats['num_tokens']['mean']:.2f}, "
                    f"Median: {phase_stats['num_tokens']['median']:.2f}, "
                    f"P80: {phase_stats['num_tokens']['p80']:.2f}"
                )
                print(
                    f"    Range: [{phase_stats['num_tokens']['min']}, {phase_stats['num_tokens']['max']}]"
                )

                print("  Time (ms):")
                print(
                    f"    Mean: {phase_stats['times']['mean']:.4f}, "
                    f"Median: {phase_stats['times']['median']:.4f}, "
                    f"P80: {phase_stats['times']['p80']:.4f}"
                )
                print(
                    f"    Range: [{phase_stats['times']['min']:.4f}, {phase_stats['times']['max']:.4f}]"
                )
                print(f"    Total: {phase_stats['times']['total']:.4f}")

                if "throughput" in phase_stats:
                    print(
                        f"  Throughput: {phase_stats['throughput']['mean_tokens_per_sec']:.2f} tokens/sec"
                    )

        print("=" * 80 + "\n")

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
