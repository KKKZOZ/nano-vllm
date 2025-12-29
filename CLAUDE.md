# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight, from-scratch implementation of vLLM in ~1,200 lines of Python. It achieves comparable or better inference speeds than vLLM while maintaining a readable codebase. The project supports prefix caching, tensor parallelism, CUDA graphs, and torch compilation.

## Development Commands

### Building and Running

This project uses uv for python venv management.

```bash
# activate the virtual environment
source ./uv/bin/activate
```

### Code Formatting
```bash
# The project uses ruff for linting with import sorting enabled
# Configuration is in pyproject.toml under [tool.ruff.lint]
ruff check .
ruff format .
```

### Model Download
```bash
# Download model weights manually
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Architecture

### Two APIs: LLM and Backend

Nano-vLLM provides two distinct APIs:

1. **LLM (High-level API)**: For batch offline inference (example.py)
   - Handles scheduling, batching, and tokenization automatically
   - vLLM-compatible interface with minor differences
   - Best for processing multiple prompts efficiently

2. **Backend (Low-level API)**: For fine-grained control (example_backend.py)
   - Direct forward() interface for token-by-token generation
   - Supports extend (multiple tokens) and rollback operations
   - Useful for speculative decoding or custom generation logic
   - Requires manual sequence ID management and tokenization

### Core Components

#### Engine Layer (`nanovllm/engine/`)
- **LLMEngine**: Main entry point for batch inference. Coordinates scheduling and model execution. The `LLM` class in `llm.py` is just an alias.
- **Backend**: Low-level API for direct forward calls with sequence management
- **Scheduler**: Implements continuous batching with preemption. Manages transitions between WAITING → RUNNING → FINISHED states. Prioritizes prefill over decode batches.
- **BlockManager**: KV cache management with prefix caching via xxhash-based block deduplication. Block size is 256 tokens (configurable). Supports allocation, deallocation, and reference counting for shared blocks.
- **ModelRunner**: Handles model execution, CUDA graph capture, tensor parallelism via torch.distributed, and KV cache allocation. Spawns worker processes for multi-GPU execution.
- **Sequence**: Tracks individual generation request state (token IDs, block table, completion status)

#### Model Layer (`nanovllm/models/`)
- Currently supports **Qwen3** models (Qwen3ForCausalLM)
- Model classes implement `packed_modules_mapping` for weight loading efficiency (QKV proj, gate-up proj fusion)
- To add support for new models:
  1. Create a new model file (e.g., `llama.py`) following the Qwen3 structure
  2. Implement the model using the layers from `nanovllm/layers/`
  3. Define `packed_modules_mapping` for efficient weight loading
  4. Update the model loader in `nanovllm/utils/loader.py`

#### Layers (`nanovllm/layers/`)
- Custom layer implementations optimized for inference:
  - **Attention**: Flash attention integration with PagedAttention for KV cache
  - **Linear**: QKVParallelLinear, RowParallelLinear, ColumnParallelLinear, MergedColumnParallelLinear for tensor parallelism
  - **RMSNorm**: Fused layernorm with residual connection
  - **RotaryEmbedding**: RoPE implementation
  - **Sampler**: Token sampling with temperature

#### Configuration (`nanovllm/config.py`)
Key parameters:
- `max_num_batched_tokens`: Maximum tokens in a batch (default: 40960)
- `max_num_seqs`: Maximum concurrent sequences (default: 512)
- `kvcache_block_size`: Block size for KV cache (default: 256, must be multiple of 256)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism (1-8)
- `enforce_eager`: Disable CUDA graph capture (useful for debugging)
- `gpu_memory_utilization`: Fraction of GPU memory for KV cache (default: 0.4)

### Tensor Parallelism

Multi-GPU execution uses `torch.multiprocessing` with `spawn` context:
- Rank 0 is the primary process; ranks 1+ are worker processes
- Communication via SharedMemory (name="nanovllm") for method calls and arguments
- Workers run a loop reading from shared memory and executing methods
- Uses NCCL backend for distributed operations
- Port configurable via `MASTER_PORT` env var (default: 2333) to support multiple instances

### CUDA Graph Support

When `enforce_eager=False`:
- ModelRunner captures CUDA graphs for different batch sizes during warmup
- Graphs are reused during inference for faster execution
- Disabled during debugging or when dynamic control flow is needed

### Prefix Caching

BlockManager implements automatic prefix caching:
- Each full block (256 tokens) is hashed using xxhash
- Hash includes prefix hash for sequence-aware deduplication
- Shared blocks use reference counting
- When allocating, checks hash table for existing blocks with same content
- `num_cached_tokens` tracks how many tokens were found in cache

### Hybrid Generator (`hybrid_generator/`)

Experimental directory containing hybrid generation strategies (speculative decoding, etc.). See `hybrid_generator/README.md` for details. This is separate from the main nano-vllm inference engine.

## File Organization

```
nanovllm/
├── __init__.py          # Exports LLM, Backend, SamplingParams
├── llm.py               # Thin wrapper around LLMEngine
├── backend.py           # Low-level API with forward() interface
├── config.py            # Configuration dataclass
├── sampling_params.py   # Sampling parameters
├── engine/
│   ├── llm_engine.py    # High-level batch inference engine
│   ├── scheduler.py     # Continuous batching scheduler
│   ├── block_manager.py # KV cache block management with prefix caching
│   ├── model_runner.py  # Model execution, CUDA graphs, tensor parallelism
│   └── sequence.py      # Sequence state tracking
├── layers/              # Custom inference-optimized layers
├── models/              # Model implementations (currently Qwen3)
└── utils/               # Utilities (model loading, logging, context management)

hybrid_generator/        # Experimental hybrid generation strategies
```

## Important Implementation Notes

- The project sets `torch._dynamo.config.recompile_limit = 64` in `__init__.py` for torch compilation
- Token IDs are managed differently between LLM and Backend APIs:
  - LLM: accepts strings or token ID lists, handles tokenization internally
  - Backend: requires manual tokenization, works with raw token IDs
- Block allocation happens lazily during scheduling (prefill phase)
- Decode phase may preempt sequences if KV cache is full
- Model weights are loaded using `nanovllm/utils/loader.py` which handles packed module mapping
- The codebase uses Python 3.10-3.12 (specified in pyproject.toml)

## Dependencies

Core dependencies (from pyproject.toml):
- torch >= 2.4.0
- triton >= 3.0.0
- transformers >= 4.51.0
- flash-attn
- xxhash (for prefix caching)
- lm-eval >= 0.4.9.2 (for evaluation)