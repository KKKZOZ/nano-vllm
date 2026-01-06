import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import logger


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.max_extend_len = config.max_extend_len
        self.rank = rank
        self.event = event

        if not dist.is_initialized():
            # Allow port configuration via environment variable to support multiple instances
            master_port = os.environ.get("MASTER_PORT", "2333")
            dist.init_process_group(
                "nccl",
                f"tcp://localhost:{master_port}",
                world_size=self.world_size,
                rank=rank,
            )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if self.config.enable_extend_cudagraph and self.extend_graphs:
                del self.extend_graphs, self.extend_graph_vars
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence(i, [0] * max_model_len) for i in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        # used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )
        logger.debug(f"KV cache block size: {block_bytes / 2**20} MB")
        config.num_kvcache_blocks = (
            int((free - peak + current) * config.gpu_memory_utilization) // block_bytes
        )
        if config.num_kvcache_blocks <= 0:
            logger.warning(
                f"Calculated num_kvcache_blocks is {config.num_kvcache_blocks}. "
                f"Free: {free / 1024**3:.2f}GB, Peak: {peak / 1024**3:.2f}GB, "
                f"Current: {current / 1024**3:.2f}GB, Util: {config.gpu_memory_utilization}"
            )
            # Fallback to a small number to avoid crash, or let it crash with better message
            # For now, let's allow it to crash if it's truly 0, but the logging helps debugging.
            # But technically, if utilization is > 0 and free is > 0, it should be positive unless block_bytes is huge.

        assert config.num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache. Free: {free / 1024**3:.2f}GB, "
            f"Block bytes: {block_bytes}"
        )
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_extend(self, seqs: list[Sequence]):
        """
        Prepare for extend operation where we append multiple tokens to existing sequences.
        Uses flash_attn_with_kvcache interface (similar to decode).
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            seqlen = len(seq)
            # Process newly added tokens (from num_cached_tokens onwards)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            context_lens.append(seqlen)  # Total sequence length (existing + new)

            if not seq.block_table:  # warmup
                continue

            # Generate slot_mapping for each new token
            for token_pos in range(seq.num_cached_tokens, seqlen):
                block_idx = token_pos // self.block_size
                in_block_offset = token_pos % self.block_size
                slot = seq.block_table[block_idx] * self.block_size + in_block_offset
                slot_mapping.append(slot)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Set context for flash_attn_with_kvcache (similar to decode)
        set_context(
            False,  # is_prefill=False to use flash_attn_with_kvcache path
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            is_extend=True,  # Flag to distinguish from decode
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        is_extend=False,
    ):
        # Use eager mode for prefill
        if is_prefill or self.enforce_eager:
            return self.model.compute_logits(self.model(input_ids, positions))

        # Extend: use CUDA graph if enabled and available
        if is_extend and self.config.enable_extend_cudagraph:
            num_tokens = input_ids.size(0)
            if num_tokens in self.extend_graphs:
                context = get_context()
                extend_vars = self.extend_graph_vars

                # Update graph input tensors
                extend_vars["input_ids"][:num_tokens] = input_ids
                extend_vars["positions"][:num_tokens] = positions
                extend_vars["slot_mapping"][:num_tokens] = context.slot_mapping
                extend_vars["context_lens"][:] = context.context_lens
                extend_vars["block_tables"][:, : context.block_tables.size(1)] = (
                    context.block_tables
                )

                # Replay graph
                graph = self.extend_graphs[num_tokens]
                graph.replay()
                return self.model.compute_logits(extend_vars["outputs"][:num_tokens])

        # Fallback to eager mode for extend if graph not available
        if is_extend:
            return self.model.compute_logits(self.model(input_ids, positions))

        # Decode: use CUDA Graph for single-token generation
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
            context.block_tables
        )
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(
        self, seqs: list[Sequence], is_prefill: bool, is_extend=False, do_sample=True
    ) -> list[int] | torch.Tensor:
        cache_hit = is_prefill and all(
            seq.num_cached_tokens == len(seq) for seq in seqs
        )
        if cache_hit:
            # Full prefix cache hit: return logits for the last token via decode path.
            input_ids, positions = self.prepare_decode(seqs)
        elif is_extend:
            input_ids, positions = self.prepare_extend(seqs)
        elif is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(
            input_ids, positions, is_prefill and not cache_hit, is_extend
        )
        reset_context()

        if do_sample:
            token_ids = (
                self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            )
            assert token_ids is not None
            return token_ids
        else:
            return logits

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        logger.info(f"Capturing CUDA graphs for batch sizes: {self.graph_bs}")
        torch.cuda.synchronize()
        free_before_decode, _ = torch.cuda.mem_get_info()
        # CUDA graph capture for decode
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        torch.cuda.synchronize()
        free_after_decode, _ = torch.cuda.mem_get_info()
        decode_graphs_mb = (free_before_decode - free_after_decode) / (1024 * 1024)
        logger.info(f"Decode CUDA graphs total: {decode_graphs_mb:.2f} MB")

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # CUDA graph capture for extend (using flash_attn_with_kvcache)
        if config.enable_extend_cudagraph:
            self.extend_graph_len = [i for i in range(1, self.max_extend_len + 1)]
            self.extend_graphs = {}
            max_extend_len = max(self.extend_graph_len)

            # Allocate tensors for extend graphs
            extend_input_ids = torch.zeros(max_extend_len, dtype=torch.int64)
            extend_positions = torch.zeros(max_extend_len, dtype=torch.int64)
            extend_slot_mapping = torch.zeros(max_extend_len, dtype=torch.int32)
            extend_context_lens = torch.zeros(1, dtype=torch.int32)
            extend_block_tables = torch.zeros(1, max_num_blocks, dtype=torch.int32)
            extend_outputs = torch.zeros(max_extend_len, hf_config.hidden_size)

            logger.info(
                f"Capturing CUDA graphs for extend lengths: 1-{max(self.extend_graph_len)}"
            )
            torch.cuda.synchronize()
            free_before_extend, _ = torch.cuda.mem_get_info()
            for length in reversed(self.extend_graph_len):
                graph = torch.cuda.CUDAGraph()

                # Set context for flash_attn_with_kvcache (similar to decode)
                set_context(
                    False,  # is_prefill=False to use flash_attn_with_kvcache
                    slot_mapping=extend_slot_mapping[:length],
                    context_lens=extend_context_lens,
                    block_tables=extend_block_tables,
                    is_extend=True,
                )

                # Warmup
                extend_outputs[:length] = self.model(
                    extend_input_ids[:length], extend_positions[:length]
                )

                # Capture
                with torch.cuda.graph(graph, self.graph_pool):
                    extend_outputs[:length] = self.model(
                        extend_input_ids[:length], extend_positions[:length]
                    )

                self.extend_graphs[length] = graph
                torch.cuda.synchronize()
                reset_context()
            torch.cuda.synchronize()
            free_after_extend, _ = torch.cuda.mem_get_info()
            extend_graphs_mb = (free_before_extend - free_after_extend) / (1024 * 1024)
            logger.info(f"Extend CUDA graphs total: {extend_graphs_mb:.2f} MB")

            self.extend_graph_vars = dict(
                input_ids=extend_input_ids,
                positions=extend_positions,
                slot_mapping=extend_slot_mapping,
                context_lens=extend_context_lens,
                block_tables=extend_block_tables,
                outputs=extend_outputs,
            )
        else:
            self.extend_graphs = {}
            self.extend_graph_len = []
