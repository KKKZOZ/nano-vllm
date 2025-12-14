# Repository Guidelines

## Project Structure & Modules
- Core code lives in `nanovllm/`: `engine/` handles scheduling, multiprocessing, and tokenization; `layers/` and `models/` implement architecture pieces; `utils/` holds helpers; `config.py` and `sampling_params.py` define runtime knobs.
- Entry points and examples are at the repo root: `example.py` (basic generation), `bench.py` and `serving_bench.py` (throughput/latency), and `assets/` for docs visuals. Packaging is described in `pyproject.toml`.

## Setup, Build, and Runtime Commands
- always run `.venv/bin/python` instead

## Coding Style & Naming
- Follow PEP 8 with 4-space indentation and type hints where practical. Use snake_case for functions/variables, CamelCase for classes (e.g., `LLMEngine`, `SamplingParams`).
- Keep modules small and composable; reuse existing abstractions (`Config`, `Sequence`, `Scheduler`, `ModelRunner`) instead of duplicating logic.
- Prefer deterministic seeds (`seed(0)`) for benchmarks and examples; keep public APIs close to vLLM semantics when feasible.

## Testing & Benchmarks
- No formal test suite exists yet; add `pytest` tests under `tests/` when contributing. Mirror current examples to cover scheduling, tokenization, and model runner behaviors with small fixtures.
- Before opening a PR, run `python bench.py` and/or `python serving_bench.py` to ensure no regression in throughput, TTFT, or TPOT; include hardware and model details when reporting numbers.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., “Add scheduler backpressure handling”). Keep commits focused and reversible.
- PRs should state the goal, key changes, testing performed (commands + hardware), and any performance deltas. Link related issues or discussions.
- Include repro steps for new features (model path, parameters used) and screenshots/plots only when they materially clarify performance or behavior.

## Security & Assets
- Do not commit model weights or large artifacts; reference download commands instead. Keep any credentials out of configs and scripts.
- Favor configurable paths for models/caches (e.g., `~/huggingface/...`) so contributors can align with their local setup.
