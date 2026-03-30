# **AI 2048: Hybrid RL + Expectimax Search**

**A Production-Grade System Bridging Deep Reinforcement Learning and Classical Search**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## **Quick Links**
- [Demo](#demo) | [Benchmarks](#performance-results) | [Installation](#installation) | [CLI Docs](#command-line-interface)

## **Overview**

This project implements a **hybrid AI agent** for the game 2048 that combines:
1.  **Deep Reinforcement Learning** (Masked PPO) for learned value estimation.
2.  **Expectimax Search** (classical game tree search) for tactical planning.
3.  **Production-optimized C++ engine** achieving major speedup over Python implementations.

The core insight: **learned value functions can replace hand-crafted heuristics** in classical search algorithms, reducing search depth requirements while maintaining strong performance.

**Key Result:** 58% win rate (reaching 2048+ tile) with depth-3 search, compared to 0% for the standalone RL policy.

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
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --depth 3
```

---

## **Performance Results**

### **Ablation Study: Search Depth Impact**

All benchmarks conducted over **100 episodes**.

| Configuration | Avg Score            | 2048+ Win Rate | Max Tile (Frequency) | Avg Moves |
|---------------|----------------------|----------------|----------------------|-----------|
| **Raw PPO Policy** | 7,995.6 ± 3502.67    | 0% | 1024 (18%) | 541 |
| **+ Expectimax (d=1)** | 5,127.32 ± 2482.23   | 0% | 1024 (4%) | 372 |
| **+ Expectimax (d=2)** | 14,014.08 ± 6496.21  | 13% | 2048 (13%) | 822 |
| **+ Expectimax (d=3)** | **26,523 ± 12749.82** | **58%** | **4096 (8%)** | 1,393 |

While the raw policy struggles to reach terminal states (2048) due to the dense reward structure and horizon effects, it learns a highly robust value function that enables the search to succeed.
<p align="center">
  <img src="data/benchmarks/depth3_expectimax/score_distribution.png" width="600" alt="Score Distribution"/>
  <br/>
  <em>Score distribution at depth 3: Bimodal peaks at 2048 (50%) and 4096 (8%) tiles</em>
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
                     Checkpoint: best_model.zip
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
- **Dynamic Cache Hit Rate:** Observed ~80% hit rate in early-game states (low entropy), stabilizing to 40-60% in complex late-game states.
- **Performance Gain:** Effectively pruning the search space by a factor of 2x-3x, allowing Depth-3 search to run within strict latency limits (<150ms/move).
---

#### **Batched Leaf Evaluation**

Instead of calling Python for every leaf node (slow), the searcher **batches all unique leaf states** and evaluates them in one forward pass:

```cpp
// Collect ALL unique leaf boards first
std::vector<Board> leaves_to_evaluate;
gather_leaves(root_board, depth, leaves_to_evaluate); // BFS traversal

// Single batch call to Python (fast)
std::vector<float> evaluations = batch_eval_func(leaves_to_evaluate);

// Cache results for tree search
std::map<Board, float> leaf_cache;
for (size_t i = 0; i < leaves_to_evaluate.size(); ++i)
    leaf_cache[leaves_to_evaluate[i]] = evaluations[i];
```

**Batch Sizes:**
- Depth 1: ~50-100 leaves
- Depth 2: ~500-1,000 leaves
- Depth 3: ~3,000-8,000 leaves

Inference Efficiency: On GPU-enabled systems, batched evaluation amortizes the Python Interpreter overhead and CUDA kernel launch costs across thousands of states, significantly reducing per-state inference latency.

---

### **2. Masked PPO Training**

#### **Invalid Action Masking**

Standard PPO wastes samples exploring invalid moves (e.g., moving left when tiles are already left-aligned). We use **action masking** to constrain the policy:

```python
# environment.py - Dynamic action masks

def action_masks(self) -> np.ndarray:
    masks = np.zeros(4, dtype=bool)
    for action in range(4):
        masks[action] = self.game.is_move_valid(action)
    return masks
```

---

#### **Reward Shaping: Log-Reward Scaling**

Instead of raw merge scores (which grow exponentially), we use **logarithmic scaling**:

```python
# reward.py - Log-scaled immediate rewards

def get_reward(merge_score: int) -> float:
    if merge_score <= 0:
        return 0.0
    return np.log2(merge_score) # e.g., merging 1024+1024 → log2(2048) = 11
```

**Why this works:**
- Raw scores: Merging 2048+2048 = 4096 (huge spike)
- Log scores: log2(4096) = 12 (smooth progression)
- Reduces variance in value function estimates

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

**Final Hyperparameters (Best Trial #108):**
- Learning Rate: `3.2e-4`
- Discount Factor ($\\gamma$): `0.998`
- GAE Lambda ($\\lambda$): `0.95`
- Entropy Coefficient: `0.005`
- Clip Range ($\\epsilon$): `0.2`

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
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --depth 3
```

Raw policy (no search)
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --no-search
```

Shallow search (faster, worse performance)
```bash
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --depth 1
```

---

### **Benchmark (Headless)**

Run large-scale performance evaluation without visualization:

```bash
uv run python scripts/benchmark.py <model_path> [OPTIONS]
```

**Arguments:**
- `model_path` (required): Path to trained model `.zip` file
- `--n_runs <int>`: Number of episodes to simulate (default: 10)
- `--depth <int>`: Expectimax search depth; 0 = raw policy (default: 0)
- `--output <name>`: Custom run name for output folder (default: auto-generated)
- `--device <str>`: Device for model inference: `cpu`, `cuda`, `auto` (default: auto)

**Examples:**

Full 100-episode benchmark with depth-3 search
```bash
uv run python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip \
  --n_runs 100 --depth 3 --output depth3_final_eval
```

Quick 10-episode test with raw policy
```bash
uv run python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip \
  --n_runs 10 --depth 0 --output raw_policy_baseline
```

CPU-only benchmark (no GPU required)
```bash
uv run python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip \
  --n_runs 50 --depth 2 --device cpu --output depth2_cpu_test
```

**Output Structure:**

```text
data/benchmarks/<run_name>/
├── results.json # Metrics + raw data (scores, tiles, steps)
└── score_distribution.png # Histogram with mean line
```

**Metrics Reported:**
- Average score ± std
- Min/max scores
- Average moves per episode
- Max tile distribution (frequency of 512, 1024, 2048, 4096, etc.)

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
├─ hybrid_ppo_v1.yaml # Standard training (200M steps, tested)
└─ resume_training.yaml # Resume training
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
uv run python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --depth 3
```

Run full benchmark suite:
```bash
uv run python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip \
  --n_runs 100 --depth 3 --output depth3_expectimax_test
```

---

## **Project Structure**

```text
├── cpp_src/                 # C++17 engine (pybind11 bindings)
│   ├── Fast2048.cpp         # LUT-based game logic
│   ├── ExpectimaxSearcher.cpp # Batched search with transposition table
│   ├── bindings.cpp         # Python ↔ C++ interface
│   └── CMakeLists.txt
├── twenty_forty_eight_ai/   # Python package
│   ├── agent/
│   │   ├── architecture.py  # Custom CNN (row/col/global heads)
│   │   └── callbacks.py     # W&B logging, checkpointing
│   ├── env/
│   │   ├── environment.py   # Gymnasium wrapper
│   │   └── reward.py        # Log-scaled reward function
│   └── utils/
│       ├── tensor_utils.py  # Board→Tensor conversion
│       └── searcher.py      # Python wrapper for C++ Expectimax
├── scripts/
│   ├── train.py             # PPO training loop
│   ├── tune.py              # Optuna hyperparameter search
│   ├── benchmark.py         # Headless evaluation
│   └── evaluate.py          # Visual evaluation
├── data/
│   ├── models/              
│   │   └── release/         # Finalized/Production models
│   └── benchmarks/          # JSON results + plots
└── configs/
    └── train/
        └── hybrid_ppo_v1.yaml # Standard training configuration
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
