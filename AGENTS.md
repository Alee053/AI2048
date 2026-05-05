# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI 2048 is a hybrid deep RL + expectimax search system for the 2048 game. It combines a Masked PPO policy (Stable-Baselines3) with a C++ expectimax searcher that uses the learned value function as a heuristic.

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Build C++ engine (must be done before running any Python code)
cd cpp_src
cmake -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build --config Release
cmake --install build --config Release
cd ..

# Training
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml

# Hyperparameter tuning
uv run python scripts/tune.py --config configs/tune/bayesian_opt_search.yaml

# Visual evaluation
uv run python scripts/evaluate.py <model_path> --depth 3

# Headless benchmark
uv run python scripts/benchmark.py <model_path> --n_runs 100 --depth 3
```

## Architecture

### Two-Stage Pipeline

1. **Training Phase (Python/Numba)**: Trains MaskedPPO with custom CNN on 32 parallel envs. Outputs `best_model.zip`.
2. **Inference Phase (C++)**: Loads model, runs batched expectimax search using Python for leaf evaluation.

### Python Package (`twenty_forty_eight_ai/`)

- `env/game.py` — Pure Python 2048 game logic (used during training)
- `env/environment.py` — Gymnasium wrapper with action masking
- `agent/architecture.py` — CustomCNN with row/col/grid pathways and tile embedding (0-16 log values)
- `utils/searcher.py` — Python wrapper exposing `searcher` C++ module

### C++ Engine (`cpp_src/`)

- `Fast2048.cpp` — LUT-based O(1) moves; precomputed tables for 65,536 row states
- `ExpectimaxSearcher.cpp` — Batched expectimax with transposition table caching
- `bindings.cpp` — pybind11 module exposing `Fast2048` and `ExpectimaxSearcher` classes

### Key Flow at Inference

```
Python board → C++ expectimax (find_best_move)
  → Gathers all leaf nodes at depth
  → Batch-evaluates via Python CNN (single forward pass)
  → Returns best move
```

The C++ module is installed to `twenty_forty_eight_ai/utils/searcher.cpython-312-x86_64-linux-gnu.so`.

## Configuration

YAML configs in `configs/train/` and `configs/tune/`. Training configs require keys: `project_name`, `run_name`, `output_dir`, `total_timesteps`, `n_envs`, `save_interval`, `features_dim`, `ppo_params`, and optionally `load_model`/`checkpoint_path` for resume.

## Notes

- Python 3.12 enforced via `pyproject.toml`
- CUDA PyTorch from custom index (`pytorch-cu130`)
- W&B for experiment tracking (login via `wandb login`)

## Documentation Policy

- Design specs and planning docs live in `docs/superpowers/specs/` but are **never committed** to git — this keeps the repo history clean
- Superpowers skill docs are gitignored alongside `docs/` and `.claude/`
- Only actual code, configs, and meaningful project documentation gets committed

## Git Commit Messages

- Use clean, concise commit messages without Co-Authored-By trailers