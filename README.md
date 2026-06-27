# **AI 2048: Hybrid RL + Expectimax Search**

**A Production-Grade System Bridging Deep Reinforcement Learning and Classical Search**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## **Quick Links**
- [Demo](#demo) | [Benchmarks](#performance-results) | [Installation](#installation) | [CLI Docs](#command-line-interface) | [Seed Sweeps](#seed-sweep-training) | [Aggregation](#post-processing-aggregation)

## **Overview**

This project implements a **hybrid AI agent** for the game 2048 that combines:
1.  **Deep Reinforcement Learning** (Masked PPO) for learned value estimation.
2.  **Expectimax Search** (classical game tree search) for tactical planning.
3.  **Production-optimized C++ engine** achieving major speedup over Python implementations.

The core insight: **learned value functions can replace hand-crafted heuristics** in classical search algorithms, reducing search depth requirements while maintaining strong performance.

**Key Result (v3, D4-augmented):** **36,275.87 ± 16,673.48** mean score (median **33,484**, range 14,820–79,864) on a 30-game depth-3 benchmark. Win rates: **100% at 1024+, 80% at 2048+, 20% at 4096+**. This is a **1.4× improvement in mean score** over the v1 release (26,523 ± 12,750 mean, 58% at 2048+). At depth 4 the same model reached **74,020** (max tile 4096) in a single-game run, finishing cleanly with no cap hits.

---

## **Demo**

<p align="center">
  <!-- Ensure assets/demo.gif exists or replace with a valid path -->
  <img src="assets/demo.gif" width="400" alt="Agent playing 2048"/>
  <br/>
  <em>
    Agent demonstrating an emergent "Wall Strategy" (keeping max tile on edge center).
    <br/>
    Unlike the human "Snake" heuristic, the agent leverages Depth-3 Expectimax to maintain stability in this higher-entropy configuration.
  </em>
</p>

**Try it yourself:**
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --depth 3
```

---

## **Performance Results**

### **Ablation Study: Search Depth Impact**

Raw-policy and depth-1/2 results use the v1 release model. Depth-3+ results are the v3 model (D4-augmented) on the same code; they are not directly comparable to the v1 rows because the model changed.

| Configuration | Model | Avg Score | 2048+ Win Rate | Max Tile (Frequency) | Notes |
|---------------|-------|-----------|----------------|----------------------|-------|
| **Raw PPO Policy** | v1 | 7,995.6 ± 3,502.67 | 0% | 1024 (18%) | 100 games |
| **+ Expectimax (d=1)** | v1 | 5,127.32 ± 2,482.23 | 0% | 1024 (4%) | 100 games |
| **+ Expectimax (d=2)** | v1 | 14,014.08 ± 6,496.21 | 13% | 2048 (13%) | 100 games |
| **+ Expectimax (d=3)** | v1 | 26,523 ± 12,749.82 | 58% | 4096 (8%) | 100 games (pre-D4 baseline) |
| **+ Expectimax (d=3)** | **v3 (D4-aug)** | **36,275.87 ± 16,673.48** (median 33,484, range 14,820–79,864) | **80% at 2048+**, 20% at 4096+, 100% at 1024+ | 4096 (20%), 2048 (60%), 1024 (20%) | 30 games, current release |
| **+ Expectimax (d=4)** | **v3 (D4-aug)** | **74,020.00** | 100% | 4096 (100%) | 1 game (n=1 sample) |

The v1→v3 jump at depth 3 is the regression fix documented in
[`docs/DEPTH3-REGRESSION-ROOT-CAUSE.md`](docs/DEPTH3-REGRESSION-ROOT-CAUSE.md):
the OLD model's value network was not invariant to the 8 D4 symmetries of the
board, and the C++ searcher's canonicalize-then-unpack path returned the
canonical form (not the search-time orientation), biasing every leaf
evaluation. The fix retrains the value network with random D4 augmentation
so that `model(canonicalize(b)) ≈ model(b)`, restoring accurate leaf
evaluations under the C++ canonicalization. See "D4 Augmentation" below.

**To reproduce the v3 depth-3 numbers** (30 games, ~3.5h on a T4 GPU):
```bash
uv run python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n_runs 30 --depth 3 --device cuda --output v3_depth3_final
```

**To verify D4 invariance of the released model** (≤30s on GPU):
```bash
uv run python scripts/check_d4_invariance.py
```

**v3 depth-3 score distribution** (30 games, the run whose numbers are in the table above):

<p align="center">
  <img src="assets/v3_depth3_score_distribution.png" width="700" alt="v3 Depth-3 Score Distribution"/>
  <br/>
  <em>30-game depth-3 score distribution: min 14,820, median 33,484, mean 36,276, max 79,864. Bimodal peaks at 2048 (60%) and 4096 (20%); 20% of games capped at 1024.</em>
</p>


---

### **Analysis: The Value Function as Heuristic**

The ablation study reveals a **critical insight** about hybrid RL systems:

#### **1. The "Shallow Search Trap" (Depth 1)**
Performance **degrades** at depth 1 (5,127 → 7,996). This suggests the value function $V(s)$ contains **local noise**. A shallow 1-step lookahead **overfits** to these noisy estimates, making worse decisions than the policy $\pi(s)$, which has learned robust action priors through training.

#### **2. The "Search as Regularization" Effect (Depth 3)**
Increasing depth to 3 acts as a **Monte Carlo averaging** process. By aggregating value estimates over thousands of leaf nodes in the search tree, Expectimax **filters out noise** in $V(s)$. This produces a **3.3x score improvement** and enables strategic play (reaching 4096 tile).

**Connection to Bayesian Optimization:** This mirrors the exploration-exploitation trade-off in GP-UCB (Krause et al., 2009). Deeper search increases sample complexity but reduces epistemic uncertainty, similar to how UCB balances mean prediction with confidence bounds.

---

## **Technical Architecture**

### **System Design Philosophy**

The project follows a **two-stage pipeline** optimized for both training efficiency and inference performance:

```text
┌─────────────────────────────────────────────────────────────┐
│                       TRAINING PHASE                        │
│ ┌──────────────┐          ┌────────────────────────────┐    │
│ │Python Engine │ ──────>  │   Masked PPO (Stable-SB3)  │    │
│ │ (Numba/JIT)  │          │     - Value Net V(s)       │    │
│ │  - LUT Moves │          │     - Policy Net π(s|a)    │    │
│ │              │          │ - Optuna Hyperparameter Opt│    │
│ └──────────────┘          └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                   │
                      Checkpoint: final_model.zip
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      INFERENCE PHASE                        │
│ ┌──────────────┐          ┌────────────────────────────┐    │
│ │  C++ Engine  │ ──────>  │    Expectimax Searcher     │    │
│ │              │          │   - Batch Leaf Evaluation  │    │
│ │              │ <──────  │ - Transposition Table Cache│    │
│ │              │   V(s)   │   - Chance Node Averaging  │    │
│ └──────────────┘          └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

### **1. High-Performance C++ Core**

#### **Lookup Table (LUT) Move System**

Instead of simulating tile movements with nested loops, all **65,536 possible row configurations** are precomputed at initialization:

```cpp
// Fast2048.cpp - O(1) move execution
void Fast2048::move(int direction) {
    for (auto &row : board) {
        int index = row_to_number(row); // Pack 4 tiles into 16-bit int
        merge_score += move_reward_LUT[index];
        row = move_row_LUT[index];      // ← O(1) table lookup
    }
}
```

**Technical Details:**
- **Row Encoding:** `index = tile[0] | (tile[1] << 4) | (tile[2] << 8) | (tile[3] << 12)`
- **LUT Size:** 3 tables × 65,536 entries = **~800KB memory**
- **Precomputation:** Happens once at startup via `init_LUT()`

This optimization is critical for Expectimax search, which evaluates **10,000+ board states per move** at depth 3.

---

#### **Expectimax with Transposition Table**

The search uses **memoization** to avoid re-evaluating identical board states:

```cpp
// ExpectimaxSearcher.cpp - Cached recursive search
float max_node_substitute(const Board& board, int depth,
                          const std::map<Board, float>& leaf_cache) {
    TranspositionKey key = {board, depth};
    if (transposition_table.count(key)) // ← Check cache
        return transposition_table[key];

    // Search all moves, cache result
    float max_value = -1e9;
    for (int move = 0; move < 4; ++move) {
        // ... Expectimax logic ...
        max_value = std::max(max_value, total_value);
    }
    transposition_table[key] = max_value;  // ← Store in cache
    return max_value;
}
```

##### Transposition Table Efficacy:
- **Dynamic Cache Hit Rate:** 20.34% across the 30-game depth-3 benchmark (rising over the episode as the persistent TT warms up); 24.34% on the single depth-4 game. The TT key is the canonical D4 form of the board, so rotated boards share entries.
- **Performance Gain:** Combined with alpha-beta-style pruning and the deferred-batching leaf eval, depth-3 search averages 134M nodes visited per game at 489,831 nodes/sec; depth-4 averages 2.45B nodes per game at 504,703 nodes/sec. The search converges cleanly (`cap_hits=0`, `moves_unresolved=0`) on every run.
---

#### **Batched Leaf Evaluation (Deferred Batching)**

The C++ searcher uses **multi-pass deferred batching** instead of the OLD gather-all-first approach. On the first search pass, leaf evaluations return `UNRESOLVED`; the searcher collects those leaf keys into a batch queue, calls Python once with the full batch, then re-runs the search with cached values. This avoids the OLD's two-pass overhead and is more friendly to large `target_batch_size` (32k leaves typical).

The C++ canonicalizes the board to a single D4 element before keying the batch, then `BoardEncoder::unpack`s back to a raw board for the model. This is safe **only if the value network is D4-invariant** — see "D4 Augmentation" below.

**Batch Sizes:**
- Depth 1: ~50-100 leaves
- Depth 2: ~500-1,000 leaves
- Depth 3: ~3,000-8,000 leaves
- Depth 4: ~50,000-100,000 leaves per game (summed across moves)

Inference Efficiency: On GPU-enabled systems, batched evaluation amortizes the Python Interpreter overhead and CUDA kernel launch costs across thousands of states, significantly reducing per-state inference latency.

---

### **2. Masked PPO Training**

#### **Invalid Action Masking**

Standard PPO wastes samples exploring invalid moves (e.g., moving left when tiles are already left-aligned). We use **action masking** to constrain the policy:

```python
# environment.py - Dynamic action masks
def action_masks(self) -> np.ndarray:
    canonical = np.array(
        [self.game.is_move_valid(act) for act in range(4)], dtype=bool
    )
    return transform_action_mask(canonical, self._current_d4)
```

The `transform_action_mask` call is the D4-augmentation hook: when training
with `d4_augment=True`, the env presents the board under a random D4
symmetry and the action mask is permuted to match. See "D4 Augmentation"
below.

---

#### **Reward Shaping: Merge + Free Cells + Snake Gradient**

The reward function in `reward.py` combines three terms, computed on the
canonical (untransformed) board so it is invariant to the D4 augmentation:

```python
# reward.py - reward = log-merge + free-cells bonus + snake-gradient bonus

def calculate_reward(board, merge_score, moved):
    if not moved:
        return -1.0
    merge_reward = np.log2(merge_score) if merge_score > 0 else 0.0
    free_cells_reward = np.sum(board == 0)
    log_board = log2_where_nonzero(board)
    snake = max(log_board * ROW_GRADIENT, log_board * COL_GRADIENT).sum()
    return MERGE_COEF * merge_reward + FREE_COEF * free_cells_reward + GRADIENT_COEF * snake
```

**Why this works:**
- **Log merge score:** reduces variance as tile values grow exponentially.
- **Free-cells bonus:** discourages filling the board (more empty cells = more moves available).
- **Snake gradient:** soft prior toward keeping the max tile in a corner with descending values along a row/column.

---

#### **Custom Neural Network Architecture**

The policy/value network uses a **specialized CNN** designed for 2048's structure:

```text
Input: board (log-normalized tiles)
       │
       ├─> Conv2D(32, kernel=2×4) ──> Row-wise patterns (e.g., [2,2,4,8])
       │
       ├─> Conv2D(32, kernel=4×2) ──> Column-wise patterns
       │
       └─> Conv2D(64, kernel=3×3) ──> Global spatial features
              │
              └─> Concatenate([row, col, global]) ──> Dense(256) ──> {Policy, Value}
```

**Why custom CNN over MLP?**
- 2048 exhibits **translational patterns** (e.g., row `[2,4,8,16]` should be recognized anywhere on board).
- CNNs share weights across positions, improving sample efficiency.

---

### **3. Systematic Hyperparameter Optimization**

All hyperparameters were tuned using **Optuna** (100+ trials, logged to Weights & Biases):

```python
# tune.py - Bayesian hyperparameter search

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.01)

    model = MaskablePPO(..., learning_rate=lr, gamma=gamma, ent_coef=ent_coef)
    model.learn(total_timesteps=5_000_000)

    # Evaluate on 50 episodes
    return evaluate_agent(model, n_episodes=50)
```

**Final Hyperparameters (v3 release, `configs/train/hybrid_ppo_v3.yaml`):**
- Learning Rate: `2.5e-4` (linear decay to 0)
- Discount Factor ($\gamma$): `0.95`
- GAE Lambda ($\lambda$): `0.95`
- Entropy Coefficient: `6.7e-6`
- Clip Range ($\epsilon$): `0.2`
- Batch Size: `4096`
- Epochs per update: `4`
- Total timesteps: `100,000,000`
- Parallel envs: `128`

---

## **Research Connections**

### Relation to Bayesian Optimization (GP-UCB, Srinivas et al. 2009)

Tuning the PPO hyperparameters for this agent is an instance of **black-box optimization**:
- Input: hyperparameter vector \(x\)
- Output: noisy objective \(f(x)\) = average 2048 score over 50 games

I used **Optuna’s Bayesian optimization** to choose hyperparameters, which is conceptually close to the **GP-UCB** framework: a surrogate model of \(f\) is updated from past trials and an acquisition rule balances exploration (trying uncertain regions) and exploitation (refining promising ones).
### Conceptual Parallel: Planning Under Uncertainty

Although Expectimax in this project does **not** implement Gaussian-process UCB, it faces a related trade-off: 
- The value function \(V(s)\) acts as a learned surrogate for long-term return.
- Expectimax search over stochastic tile spawns reduces uncertainty by averaging over possible futures.

This is philosophically similar to Bayesian optimization’s use of surrogate models and uncertainty-aware acquisition, but without explicit confidence bounds or regret guarantees.

---

## **Command-Line Interface**

All scripts support the `--help` flag for detailed usage information. Below are common workflows:

### **Training**

Train a new agent from scratch or resume from a checkpoint:

```bash
uv run python scripts/train.py --config <path_to_yaml>
```

**Required Config Keys (YAML):**

```yaml
project_name: "2048-hybrid-ai" # W&B project name
run_name: "ppo-expectimax-v1" # Unique run identifier
output_dir: "data/" # Base directory for outputs
total_timesteps: 200_000_000 # Total training steps
n_envs: 32 # Parallel environments
save_interval: 5_000_000 # Checkpoint frequency (steps)
features_dim: 256 # CNN feature extractor output size

# PPO Hyperparameters
ppo_params:
  learning_rate:
    type: "linear_decay"
    initial_value: 0.0003
  gamma: 0.998 # Discount factor
  gae_lambda: 0.95 # GAE parameter
  clip_range: 0.2 # PPO clip range
  ent_coef: 0.005 # Entropy coefficient

# Resume Training (optional)
load_model: false # Set to true to resume
checkpoint_path: null # Path to .zip checkpoint
```

**Example:**

Fresh training
```bash
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml
```

Resume from checkpoint
```bash
uv run python scripts/train.py --config configs/train/resume_training.yaml
```

**Seed Sweep Training:**

Run multiple seeds sequentially for statistical robustness:
```bash
# Dry run to preview what would be launched
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml \
  --seed-sweep 3 --dry-run

# Launch 5-seed sweep (sequential)
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml \
  --seed-sweep 5

# Resume a failed sweep (skips completed seeds, re-runs failed/pending)
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml \
  --seed-sweep 5 --resume-sweep
```

**Arguments:**
- `--seed <int>`: Set a single fixed seed
- `--seed-sweep <N>`: Launch N sequential runs with seeds 0..N-1
- `--resume-sweep`: Resume interrupted sweep (skip completed, re-run failed/pending)
- `--dry-run`: Print sweep plan without launching any jobs

**Output Structure:**
```
data/models/<run_name>/
├── sweep_status.json        # Tracks seed completion status
├── seed_0/
│   └── final_model.zip
├── seed_1/
│   └── final_model.zip
└── ...
```

Each seed run's W&B run name is `<run_name>-seed<N>` for easy filtering.

---

### **Hyperparameter Tuning**

Run Bayesian optimization over hyperparameter search space:

```bash
uv run python scripts/tune.py --config <path_to_tune_config>
```

**Required Config Keys (YAML):**

```yaml
project_name: "2048-optuna-study"
study_name: "ppo-hp-search-v1"
db_path: "data/optuna/study.db" # SQLite storage
timeout_hours: 48 # Max study duration

# Search space definitions
ppo_search_space:
  learning_rate:
    type: "float"
    low: 0.00001
    high: 0.001
    log: true
  gamma:
    type: "float"
    low: 0.95
    high: 0.999
  ent_coef:
    type: "float"
    low: 0.0
    high: 0.01

# Trial configuration
trial:
  n_envs: 16
  total_timesteps: 5_000_000
  report_freq: 100_000

# Pruner settings
pruner:
  n_startup_trials: 5
  n_warmup_steps: 500_000
```

**Resumable Studies:**
Optuna automatically resumes from the SQLite database if `load_if_exists=True` (default).

---

### **Evaluation (Visual)**

Watch the agent play with interactive pygame visualization:

```bash
uv run python scripts/evaluate.py <model_path> [OPTIONS]
```

**Arguments:**
- `model_path` (required): Path to trained model `.zip` file
- `--no-search`: Disable Expectimax search (use raw PPO policy)
- `--depth <int>`: Search depth for Expectimax (default: 3)

**Examples:**

Depth-3 Expectimax (recommended)
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --depth 3
```

Raw policy (no search)
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --no-search
```

Shallow search (faster, worse performance)
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --depth 1
```

---

### **Benchmark Harness**

Run large-scale, paper-grade evaluation without visualization. The harness emits structured CSV + JSON suitable for downstream analysis (`aggregate.py`, pandas, paper figures).

**Invocation:**

```bash
uv run python -m scripts.benchmark <model_path> [OPTIONS]
```

The `-m scripts.benchmark` form is required because `scripts/` is a package; running `python scripts/benchmark.py` directly will fail with `ModuleNotFoundError: No module named 'scripts'`.

**Arguments:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `model_path` | str | (required unless `--model-dir`) | Path to trained model `.zip` |
| `--n-runs` | int | `100` | Number of episodes |
| `--depth` | int | `0` | Expectimax search depth; `0` = raw policy |
| `--output` | str | `run_<timestamp>` | Folder name under `data/benchmarks/` |
| `--device` | `{cpu,cuda,auto}` | `auto` | Inference device |
| `--workers` | int | `1` | Number of subprocess workers |
| `--log-moves` | flag | off | Write `moves.csv` with one row per player move |
| `--yes-large-move-log` | flag | off | Required if estimated move-log rows > 5M |
| `--base-eval-seed` | int | random | Root seed for deterministic per-episode eval seeds |
| `--train-seed` | int | none | Recorded in `config.json` for sweep runs |
| `--model-version` | str | none | Free-form version label, recorded in `config.json` |
| `--model-dir` | str | none | Directory of `seed_N/` subdirs (multi-seed; placeholder in current build) |
| `--parallel` | flag | off | Parallel across seeds (multi-seed mode only) |

**Quick examples:**

```bash
# Paper-grade single-worker, depth-3, 100 episodes
uv run python -m scripts.benchmark \
  data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n-runs 100 --depth 3 --workers 1 --device cuda \
  --output v3_depth3_final --base-eval-seed 0

# Throughput-mode: 8 CPU workers, depth-3
uv run python -m scripts.benchmark \
  data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n-runs 200 --depth 3 --workers 8 --device cpu \
  --output v3_depth3_throughput --base-eval-seed 0

# Raw-policy baseline (no search)
uv run python -m scripts.benchmark \
  data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n-runs 100 --depth 0 --workers 1 \
  --output v3_raw --base-eval-seed 0

# Per-move log for downstream analysis
uv run python -m scripts.benchmark \
  data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n-runs 50 --depth 3 --workers 1 \
  --log-moves --output v3_depth3_moves
```

**Operational recommendations:**

| Goal | Recommended |
|---|---|
| Paper-grade single-GPU latency, reproducible | `--device cuda --workers 1` |
| Experimental: parallel GPU inference | `--device cuda --workers 2` (only if GPU memory fits two model copies) |
| Maximum throughput on a CPU box | `--device cpu --workers 4-8` |

#### Output Structure

Every run produces four files in `data/benchmarks/<run_name>/`:

```
data/benchmarks/<run_name>/
├── config.json    # Run provenance + config (written at start, updated at end)
├── episodes.csv   # One row per completed episode
├── moves.csv      # One row per player move (only when --log-moves)
└── summary.json   # Aggregate metrics + status
```

The harness writes `episodes.csv` and `moves.csv` incrementally with `flush()` after each row — a `kill -9` mid-run leaves a usable partial output plus a `config.json` with `interrupted=true` and `status="interrupted"`.

#### `config.json` schema

```json
{
  "benchmark_schema_version": "1.0.0",
  "run_id": "uuid4...",
  "run_name": "v3_depth3_final",
  "model_path": "data/models/release/Hybrid-PPO-Expectimax-v3.zip",
  "model_md5": "fab18d67...",
  "model_version": "v3",
  "train_seed": null,
  "env_seed_base": 12345,
  "n_runs": 100,
  "n_workers": 4,
  "device": "cuda",
  "cuda_device_name": "NVIDIA RTX 4090",
  "cuda_runtime": "13.0",
  "depth": 3,
  "use_expectimax": true,
  "log_moves": false,
  "base_eval_seed": 12345,
  "eval_seed_strategy": "deterministic-offset",
  "git_commit": "abc1234",
  "git_dirty": false,
  "started_at_iso": "2026-06-27T12:34:56Z",
  "finished_at_iso": "2026-06-27T12:41:48Z",
  "total_wall_time_s": 412.7,
  "interrupted": false,
  "status": "completed"
}
```

Field semantics:

- `benchmark_schema_version`: semver. `aggregate.py` accepts same-major; rejects others.
- `env_seed_base`: seed for Python/numpy RNG used by `Fast2048.generate_random()`.
- `eval_seed_strategy`: `"deterministic-offset"` (master assigns `eval_seed = env_seed_base + episode_idx`) or `"random"` (when `base_eval_seed` is unset).
- `total_wall_time_s`: full run wall-clock (includes worker spawn + summary write).
- `status`: `"completed"` | `"interrupted"` | `"failed"`.
- `interrupted`: `true` only on `SIGINT`/`SIGTERM`.

#### `episodes.csv` columns (43 fields)

The column list lives in `scripts/benchmark_io.py:EPISODE_COLUMNS` — single source of truth for both the writer (`benchmark.py`) and the consumer (`aggregate.py`). Adding a column means updating that list AND the `EpisodeResult` dataclass together.

Per-episode fields:

| Field | Notes |
|---|---|
| `schema_version`, `run_id`, `episode_idx`, `worker_id` | Identifiers |
| `train_seed`, `eval_seed` | Seeds (reproducibility) |
| `requested_depth`, `effective_depth`, `use_expectimax` | Search config (forward-compat: `effective_depth` will diverge from `requested_depth` if iterative deepening lands) |
| `score`, `max_tile`, `max_log_tile`, `steps` | Outcome |
| `episode_time_s`, `mean_move_time_ms`, `median_move_time_ms`, `p95_move_time_ms`, `max_move_time_ms` | Wall time |
| `termination_reason` | `"board_full"` (today) or `"max_steps"` (reserved) |
| `win_1024`, `win_2048`, `win_4096`, `win_8192` | Boolean thresholds |
| `total_think_ms`, `total_nodes`, `total_batches` | C++ search think-time aggregates |
| `total_tt_lookups`, `total_tt_hits`, `total_tt_collisions`, `total_tt_same_key_overwrites` | Transposition table |
| `total_moves_resolved`, `total_moves_unresolved`, `total_cap_hits` | Resolution + iteration-cap hits |
| `total_alpha_beta_cuts`, `total_chance_nodes`, `total_max_nodes` | Search internals |
| `mean_chance_value` | Average chance-node value |
| `mean_empty_cells`, `min_empty_cells`, `mean_merge_score` | Board-state distributions |
| `mean_nps`, `mean_tt_hit_rate`, `mean_nodes_per_batch_call` | Derived rates |

#### `moves.csv` columns (29 fields)

Opt-in via `--log-moves`. One row per player move. Useful for paper figures on board evolution, tile-spawn distributions, and search-time histograms. **Can produce GB-scale files** — use `--log-moves` only when you need per-move data.

All board-snapshot fields (`board_state`, `canonical_board_hash`, `empty_cells_before`, `max_tile_before`, `max_log_tile_before`, `n_legal_actions`, `score_*`) come from the **same pre-action board snapshot**. The row describes the decision point, not the post-decision state.

| Notable field | Notes |
|---|---|
| `board_state` | 16 comma-separated log-tile values in row-major order; `0`=empty, `11`=2048-tile, `16`=65536-tile |
| `canonical_board_hash` | `BoardEncoder::canonicalize` uint64 (D4-canonical form), base-10 string |
| `score_up/right/down/left` | C++ root-move scores (or `NaN` in raw-policy mode) |
| `move_time_ms` | Wall-time around the full move (search + env step) |
| `think_ms` | Just the C++ `find_best_move` time |

`--log-moves` writes a single warning + estimated row count and disk usage before any worker spawns. If the estimate exceeds 5,000,000 rows, the run refuses to start unless `--yes-large-move-log` is passed.

#### `summary.json`

Aggregate metrics for quick inspection. Mirrors the shape of the old `results.json["metrics"]` block plus the new search-mode fields, win-rate thresholds (1024/2048/4096/8192), and a max-tile distribution.

```json
{
  "benchmark_schema_version": "1.0.0",
  "status": "completed",
  "n_completed": 100,
  "n_runs_requested": 100,
  "config": { "...subset of config.json...": "..." },
  "metrics": {
    "avg_score": 24320.5, "std_score": 8921.3, "min_score": 8400, "max_score": 51200,
    "median_score": 23456, "p25_score": 18432, "p75_score": 31204,
    "avg_steps": 487.2, "min_steps": 312, "max_steps": 901, "median_steps": 478,
    "total_time_s": 410.2, "total_wall_time_s": 412.7,
    "avg_time_per_game_s": 4.10, "games_per_sec": 0.244,
    "avg_think_ms": 38.7, "avg_nodes_visited": 1124.5, "avg_batches_eval": 3.2,
    "avg_nodes_per_sec": 29040.0, "avg_tt_hit_rate": 12.4,
    "avg_tt_collisions": 0.8, "avg_tt_same_key_overwrites": 0.1,
    "avg_moves_resolved": 3.85, "avg_moves_unresolved": 0.15,
    "avg_cap_hits": 0.0, "avg_alpha_beta_cuts": 14.2,
    "avg_chance_nodes": 287.4, "avg_max_nodes": 145.6,
    "avg_chance_value": 0.087,
    "score_ci95_low": 22612.0, "score_ci95_high": 26029.0
  },
  "win_rates": {
    "win_rate_1024+": 1.0, "win_rate_2048+": 0.82,
    "win_rate_4096+": 0.18, "win_rate_8192+": 0.02
  },
  "max_tile_dist": {
    "1024": 0, "2048": 16, "4096": 82, "8192": 2
  }
}
```

`total_time_s` excludes worker-spawn + summary-write overhead. `total_wall_time_s` includes everything.

#### Parallel execution model

The harness uses `multiprocessing.Process` with the `spawn` start method (CUDA-safe — PyTorch explicitly warns against `fork` after CUDA init). Workers are long-lived; the master owns a result queue and a status queue.

Seed assignment is **worker-count-invariant**: master precomputes `seeds = [env_seed_base + i for i in range(n_runs)]` and chunks them contiguously to workers. `episode_idx=i` always sees `eval_seed = env_seed_base + i` regardless of `--workers`.

```text
seeds = [0, 1, 2, 3, 4, 5, 6, 7]

--workers 1:  worker 0 = [0, 1, 2, 3, 4, 5, 6, 7]
--workers 2:  worker 0 = [0, 1, 2, 3]    worker 1 = [4, 5, 6, 7]
--workers 4:  worker 0 = [0, 1]          worker 1 = [2, 3]
              worker 2 = [4, 5]          worker 3 = [6, 7]
```

Per-worker RNG is seeded once at process start (`np.random.seed(env_seed_base + worker_id * 10_000)`). The C++ searcher's chance-node evaluation is **deterministic** — it enumerates every empty cell with both tile values 2 and 4 and computes the exact expected value, so no C++ RNG seeding is required (verified by `tests/unit/test_searcher_determinism.py`).

**Important reproducibility caveat:** scores are reproducible across runs at the **same worker count** but may differ slightly between `--workers 1` and `--workers 2`. The spec classifies score as "match approximately across worker counts" because per-worker `np.random` state ordering interleaves differently. The fields that are guaranteed worker-count-invariant are `eval_seed`, `episode_idx`, `use_expectimax`, `requested_depth`, `effective_depth`, `schema_version`, `run_id`, `worker_id`, `train_seed`, `termination_reason`, and `steps` distribution shape.

#### Interrupt + crash handling

| Signal / event | Behavior |
|---|---|
| `Ctrl-C` (SIGINT) | Master sets `stop_event`, drains in-flight queue non-blocking, joins workers (10s timeout then `terminate()`), writes partial `summary.json` with `status="interrupted"` |
| `SIGTERM` | Funneled through `KeyboardInterrupt` handler; same behavior as SIGINT |
| Worker exception | Worker posts `{status: "failed", error: traceback}` to status queue and re-raises; master marks `status="failed"` and exits non-zero |
| `kill -9` on master | Output is crash-safe: `episodes.csv`/`moves.csv` flush after each row; `config.json` is written at start so provenance survives |

Only episodes that were fully returned to the result queue produce rows. A worker's in-progress episode (not yet posted) is dropped on interrupt — by design, since partial `EpisodeResult` objects cannot be safely serialized.

#### Logging invariance

Enabling `--log-moves` does NOT change `episodes.csv` content. This is enforced by `test_log_moves_off_vs_on_produces_identical_episodes_csv` in `tests/integration/test_benchmark_csv.py`. Adding a non-deterministic column (e.g. wall-clock timestamp) to `episodes.csv` will fail this test.

---

### **Aggregation (`scripts/aggregate.py`)**

Consume `episodes.csv` outputs from one or more runs to produce paper-grade summary statistics and figures.

```bash
# Aggregate all depth results for a sweep
uv run python -m scripts.aggregate.py data/benchmarks/ --sweep sweep-v1

# Single win-threshold focus
uv run python -m scripts.aggregate.py data/benchmarks/ --sweep sweep-v1 --win-threshold 4096

# Re-process historical JSON runs
uv run python -m scripts.aggregate.py data/benchmarks/ --sweep v3_depth3_final --legacy
```

**Discovery convention:** the folder under `data/benchmarks/` must match `{sweep_name}_depth{N}` for `aggregate.py` to find it. This is enforced by `--output` naming in `benchmark.py`.

**Schema-version safety:** by default, `aggregate.py` walks each run folder, reads `config.json`, and refuses to consume a run whose `benchmark_schema_version` major differs from `1`. To override (e.g. reprocess very old runs), pass `--legacy`.

**Arguments:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `benchmark_dir` | str | (required) | Root folder containing `{sweep_name}_depth*` subfolders |
| `--sweep` | str | (required) | Sweep name to filter on |
| `--win-threshold` | int | (all) | Report a single win threshold instead of 1024/2048/4096/8192 |
| `--output` | str | `benchmark_dir` | Override output directory |
| `--legacy` | flag | off | Read legacy `results_seed_N.json` files instead of the new CSV layout |

**Output:**

```
<output_dir>/
├── summary.csv               # Per-seed + aggregate rows with all metrics
├── cross_depth_ci_table.csv  # Depth comparison with 95% confidence intervals
└── paper_figures/
    ├── violin_score_depth{N}.png
    ├── bar_winrate_depth{N}.png
    └── heatmap_max_tile.png
```

**`summary.csv` columns** (per-seed + aggregate row):

- `sweep_name`, `depth`, `seed`
- `avg_score`, `std_score`, `min_score`, `max_score`, `avg_steps`
- `win_rate_1024+`, `win_rate_2048+`, `win_rate_4096+`, `win_rate_8192+`
- `max_tile_eq_1024_pct`, `max_tile_eq_2048_pct`, `max_tile_eq_4096_pct`, `max_tile_eq_8192_pct`
- Search-mode metrics (when present): `avg_think_ms`, `avg_nodes_visited`, `avg_batches_eval`, `avg_nodes_per_sec`, `avg_tt_hit_rate`, `avg_tt_collisions`, `avg_tt_same_key_overwrites`, `avg_moves_resolved`, `avg_moves_unresolved`, `avg_cap_hits`, `avg_alpha_beta_cuts`, `avg_chance_nodes`, `avg_max_nodes`, `avg_chance_value`

---

### **Multi-Seed Benchmarking (status: placeholder)**

The `--model-dir` flag is wired through the CLI but `scripts/benchmark_multi_seed.py` is currently a stub that returns exit code 1 with a clear error. Multi-seed sweep runs are supported via repeated single-model invocations:

```bash
for seed_dir in data/models/sweep-v1/seed_*; do
  seed_name=$(basename "$seed_dir")
  uv run python -m scripts.benchmark "$seed_dir/final_model.zip" \
    --n-runs 100 --depth 3 --workers 1 \
    --output "sweep-v1_depth3/${seed_name}" \
    --base-eval-seed 0
done

# Then aggregate
uv run python -m scripts.aggregate data/benchmarks/sweep-v1_depth3 --sweep sweep-v1_depth3
```

The discovery convention `{sweep_name}_depth{N}` still applies when constructing the output paths.

---

### **Performance Profiling**

Use built-in W&B logging to track training metrics in real-time:

Training automatically logs to W&B
```bash
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml
```

View dashboard
```bash
wandb login
# Navigate to your W&B project URL
```

**Logged Metrics:**
- Episode reward (mean, min, max)
- Episode length
- Value loss
- Policy loss
- Entropy coefficient
- Learning rate schedule
- Custom 2048 metrics (max tile reached, merge efficiency)

---

### **Config File Templates**

Example config is provided in `configs/train/`:

```text
configs/train/
├─ hybrid_ppo_v1.yaml     # Original training (200M steps, v1 model)
├─ hybrid_ppo_v2_sweep.yaml # Hyperparameter sweep
├─ hybrid_ppo_v3.yaml     # D4-augmented training (100M steps, v3 model — current release)
└─ resume_training.yaml   # Resume from a checkpoint
configs/tune/
└─ bayesian_opt_search.yaml # Optuna search space
```

To create a custom config:

```bash
cp configs/train/hybrid_ppo_v1.yaml configs/train/my_experiment.yaml
# Edit my_experiment.yaml with your hyperparameters
uv run python scripts/train.py --config configs/train/my_experiment.yaml
```

## **Reproducibility**

### **System Requirements**
- **CPU:** x86-64 with AVX2 support (for fast bitboard operations)
- **GPU:** NVIDIA GPU with CUDA 11.8+ (CUDA 13 recommended)
- **RAM:** 16GB minimum (8GB for training, 4GB for inference, 4GB OS overhead)
- **Storage:** 5GB (models, logs, benchmark data)

### **Installation**
> **Shell note:** Commands below use a POSIX-style shell syntax. On Windows, you can run them from Git Bash, WSL, or adapt the `$(...)` substitution to PowerShell.


1. **Clone repository:**
   ```bash
   git clone https://github.com/Alee053/AI2048.git
   cd AI2048
   ```

2. Install Python dependencies:
   ```bash
    uv sync
   ```
   
3. Build C++ engine:
   ```bash
    cd cpp_src
    cmake -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
    cmake --build build --config Release
    cmake --install build --config Release
    cd ..
   ```
   
**Platform Notes:**
- **Windows:** Requires `--config Release` flag
- **Linux/macOS:** `--config` flag is optional (can omit)
- **CMake 3.15+:** Required for multi-config generator support

### **Quick Start**

Train a new agent (200M steps, 44 hours on a T4 GPU):
```bash
uv run python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml
```

Evaluate with visualization:
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --depth 3
```

Run full benchmark suite (paper-grade):
```bash
uv run python -m scripts.benchmark data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n-runs 100 --depth 3 --workers 1 --output depth3_expectimax_test
```

Throughput-mode benchmark (CPU, 8 workers):
```bash
uv run python -m scripts.benchmark data/models/release/Hybrid-PPO-Expectimax-v3.zip \
  --n-runs 200 --depth 3 --workers 8 --device cpu --output depth3_throughput
```

Aggregate a sweep's results into summary.csv + figures:
```bash
uv run python -m scripts.aggregate data/benchmarks/ --sweep sweep-v1
```

---

## **D4 Augmentation**

The C++ searcher's `BoardEncoder::canonicalize` is a positional optimisation
that lets the transposition table share entries between boards that are
rotations or reflections of each other. This is sound **only if the value
network is D4-invariant** (i.e. it returns the same value for all 8
symmetries of any given board). If the model is not invariant, the
canonical form the C++ uses for batch-eval differs from the
search-time orientation, and every leaf evaluation is biased.

`Game2048Env` accepts an opt-in `d4_augment=True` flag that, on every
`reset()` and `step()`, presents the board under a uniformly random D4
symmetry and inverse-permutes the agent's action:

```python
# twenty_forty_eight_ai/env/d4_transforms.py
ACTION_TO_CANONICAL = np.array([
    [0, 1, 2, 3],   # ID
    [3, 0, 1, 2],   # ROT90_CW
    [2, 3, 0, 1],   # ROT180
    [1, 2, 3, 0],   # ROT270_CW
    [0, 3, 2, 1],   # REFLECT_H
    [2, 1, 0, 3],   # REFLECT_V
    [3, 2, 1, 0],   # TRANSPOSE
    [1, 0, 3, 2],   # ANTI_TRANSPOSE
], dtype=np.int64)
```

`scripts/train.py`, `scripts/profile_train.py`, and `scripts/tune.py` set
`d4_augment=True` by default via `env_kwargs`, so the model learns
invariance without per-script configuration. Benchmark, evaluate, and
visualizer paths are untouched (the env defaults to `d4_augment=False`).

**Verify on the released v3 model:**

```bash
uv run python scripts/check_d4_invariance.py
```

On 100 random mid-game boards the released v3 model has mean abs diff
~0.35 and max diff ~1.65 across the 7 non-identity D4 elements. The OLD
v1 release (pre-augmentation) had mean ~1.0, max ~6.0 on the same boards
— a 3-4× improvement in D4 invariance. The 0.01 threshold from the
regression doc is aspirational; the CustomCNN is not rotation-equivariant
by design, so 100M steps gets the model close but not perfect. The
residual error is small enough that the C++ search still picks strong
moves (mean 36,276 at depth 3, vs the OLD's 26,523).

---

## **Project Structure**

```text
├── cpp_src/                       # C++17 engine (pybind11 bindings)
│   ├── Fast2048.cpp               # LUT-based game logic
│   ├── ExpectimaxSearcher.cpp     # Multi-pass deferred-batching searcher + TT
│   ├── BoardEncoder.cpp           # 16-bit pack/unpack/canonicalize
│   ├── bindings.cpp               # Python ↔ C++ interface
│   └── CMakeLists.txt
├── twenty_forty_eight_ai/         # Python package
│   ├── agent/
│   │   ├── architecture.py        # Custom CNN (row/col/global heads)
│   │   └── callbacks.py           # W&B logging, checkpointing
│   ├── env/
│   │   ├── environment.py         # Gymnasium wrapper (D4-augment opt-in)
│   │   ├── d4_transforms.py       # D4 symmetries + action-permutation table
│   │   ├── game.py                # Fast2048 (LUT-based)
│   │   └── reward.py              # Merge + free-cells + snake-gradient
│   ├── evaluation/
│   │   └── benchmarker.py         # Benchmarker class (runs episodes, returns EpisodeResult)
│   └── utils/
│       ├── tensor_utils.py        # Board→Tensor conversion
│       └── searcher.py            # Python wrapper for C++ Expectimax
├── scripts/
│   ├── train.py                   # PPO training (D4-augment on by default)
│   ├── tune.py                    # Optuna hyperparameter search
│   ├── benchmark.py               # CLI entry point (thin wrapper over benchmark_runner)
│   ├── benchmark_io.py            # Schema, dataclasses, CSVWriter (single source of truth)
│   ├── benchmark_runner.py        # Master process: spawn workers, drain queues, write outputs
│   ├── benchmark_worker.py        # run_worker subprocess function
│   ├── benchmark_summary.py       # compute_summary_from_rows
│   ├── benchmark_multi_seed.py    # Multi-seed stub (placeholder)
│   ├── aggregate.py               # Post-processing aggregator for sweeps
│   ├── evaluate.py                # Visual evaluation (pygame)
│   ├── profile_train.py           # Training profile run
│   └── check_d4_invariance.py     # D4 invariance check for the value net
├── tests/
│   ├── unit/                      # Fast unit tests (no subprocess)
│   │   ├── test_benchmark_io.py   # Schema, dataclasses, CSVWriter
│   │   ├── test_benchmarker.py    # Benchmarker class (raw-policy + search)
│   │   ├── test_benchmark_worker.py # run_worker subprocess
│   │   └── test_searcher_determinism.py
│   ├── integration/               # End-to-end CLI tests (require production model)
│   │   └── test_benchmark_csv.py
│   ├── test_d4_transforms.py
│   ├── test_depth4_convergence.py
│   ├── test_persistent_tt.py
│   ├── test_transposition_table.py
│   ├── test_board_encoder.py
│   ├── test_searcher_wrapper.py
│   ├── test_seed_utils.py
│   ├── test_sparkline.py
│   ├── test_visualizer_config.py
│   └── stress_depth4_real.py      # Real-model depth-4 stress test
├── data/
│   ├── models/
│   │   └── release/
│   │       └── Hybrid-PPO-Expectimax-v3.zip   # D4-augmented release
│   └── benchmarks/                # (gitignored; populated by benchmark.py)
├── configs/
│   ├── train/
│   │   ├── hybrid_ppo_v1.yaml     # v1 (no D4 aug)
│   │   ├── hybrid_ppo_v2_sweep.yaml
│   │   ├── hybrid_ppo_v3.yaml     # v3 (D4-augmented, current release)
│   │   └── resume_training.yaml
│   └── tune/
│       └── bayesian_opt_search.yaml
├── docs/
│   ├── DEPTH3-REGRESSION-ROOT-CAUSE.md  # Diagnostic + retraining plan
│   └── TODO-training-fixes.md
└── twenty_forty_eight_ai.egg-info/        # build artifact (gitignored)
```

---

## **Future Work**

### **Directions Aligned with This Project:**

- **Distributional Value Networks:** Replace point-estimate $V(s)$ with a distribution (e.g., via quantile regression or ensembles). Use variance to implement **risk-sensitive Expectimax**, penalizing high-uncertainty leaf nodes.

- **AlphaZero-Style MCTS:** Replace fixed-depth Expectimax with **Monte Carlo Tree Search** guided by the learned policy $\pi(s)$, enabling adaptive depth and more efficient exploration of promising branches.

- **Transfer to Other Stochastic Games:** Apply the hybrid RL + search framework to games like **Threes**, **2048 variants** (5×5 grid, different merge rules), or **slot-based puzzle games** with similar uncertainty structures.

### **Connections to Broader Research (Krause et al.):**

- **Bayesian Hyperparameter Optimization:** Extend the Optuna-based tuning framework to jointly optimize **RL hyperparameters** and **search hyperparameters** (depth, pruning thresholds) in a single study.

### **Not Pursued (Out of Scope):**

- Safe RL for physical systems (no hardware risk in 2048)
- Lyapunov-based stability guarantees (game has no "crash" states requiring formal safety)
---

## **Citation**

If you use this code in your research, please cite:

```bibtex
@misc{castro2025-2048hybrid,
  author = {Castro, Alejandro},
  title = {AI 2048: Hybrid Reinforcement Learning with Expectimax Search},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Alee053/AI2048}
}
```

---

## **Acknowledgments**

- **Stable-Baselines3** for Masked PPO implementation.
- **Optuna** for Bayesian hyperparameter optimization.
- **Weights & Biases** for experiment tracking.
- **pybind11** for seamless C++↔Python integration.

---

## **License**

MIT License. See `LICENSE` for details.
