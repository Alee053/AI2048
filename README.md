# AI 2048 Agent

This repository contains a high-performance AI agent for the game 2048. It combines a fast C++ game engine with a deep reinforcement learning agent trained using modern RL techniques. The agent uses its learned knowledge to guide a powerful Expectimax search, achieving very high scores.

## Features

- **High-Performance C++ Core:** The game logic and Expectimax searcher are implemented in C++ for maximum speed.
- **Python/Pybind11 Integration:** The C++ core is exposed to Python using pybind11, allowing for a fast and flexible environment for the AI agent.
- **Advanced RL Agent:** The agent uses a custom "2048-Aware" CNN architecture and is trained with MaskablePPO from [Stable Baselines3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib), which masks invalid moves.
- **PPO-Guided Expectimax Search:** During evaluation, the agent's learned value function is used to guide a batched Expectimax search, dramatically improving tactical play beyond the raw policy network's capability.
- **Config-Driven Training:** Training runs are managed via simple YAML configuration files, making it easy to define, track, and reproduce experiments.
- **Wandb Integration:** Automatically logs all training metrics, configurations, and model checkpoints to Weights & Biases for easy monitoring.

## Architecture Overview

The project is split into two main parts:

1.  **C++ Core (`cpp_src`):**
    - `Fast2048`: A highly optimized class that manages the 2048 game state and logic.
    - `ExpectimaxSearcher`: Implements a batched Expectimax search algorithm. It uses the RL agent's value function to evaluate future board states.
    - `bindings.cpp`: Exposes the C++ classes to Python as a callable module named `searcher`.

2.  **Python AI Framework (`twenty_forty_eight_ai`):**
    - **`env`**: A custom `gymnasium` environment (`Game2048Env`) that wraps the C++ game engine, serving as the interface for the RL agent.
    - **`agent`**:
        - `architecture.py`: Defines the `CustomCNN`, a novel neural network architecture with parallel pathways to analyze rows, columns, and local grid patterns after converting tile values to a rich embedding space.
        - `callbacks.py`: A custom callback for logging detailed metrics to `wandb`.
    - **`scripts`**: Entry points for `train.py`, `evaluate.py`, and `tune.py`.

## Setup and Installation

### Prerequisites

- A C++ compiler (e.g., MSVC on Windows, GCC on Linux, Clang on macOS)
- CMake (version 3.15 or higher)
- Python 3.10+

### 1. Build the C++ Module

The C++ core must be compiled into a Python module.

```bash
# Create a build directory
cd cpp_src
cmake -B build .

# Compile the module
cmake --build build --config Release
```

This will compile the C++ code and place the resulting Python module (`searcher.pyd` or `searcher.so`) into the `twenty_forty_eight_ai/utils/` directory, where the Python code expects it.

### 2. Install Python Dependencies

Install all the required Python packages using pip.

```bash
pip install -r requirements.txt
```

## How to Use

### Training a New Agent

Training is handled by `scripts/train.py` and is configured via a YAML file.

1.  **Configure the Run:** Create or edit a YAML file in the `configs/train/` directory. You can use `NewArch-GradReward-v1.yaml` as a template. Set the `run_name` and adjust hyperparameters as needed.
2.  **Start Training:** Launch the script with the path to your config file.

```bash
python scripts/train.py --config configs/train/NewArch-GradReward-v1.yaml
```

Training progress and metrics will be logged to [Weights & Biases](https://wandb.ai). Models will be saved periodically in the `data/models/<run_name>/` directory.

### Evaluating a Trained Agent

Use `scripts/evaluate.py` to watch a trained agent play the game using a Pygame visualizer.

The script can run the agent in two modes:
- **PPO + Expectimax (Default):** The agent uses its value function to guide a deep Expectimax search. This is the strongest mode of play.
- **Raw PPO Policy (No Search):** The agent makes decisions based on the direct output of its policy network. This is faster but less powerful.

**Usage Examples:**

```bash
# Evaluate a model using the default PPO-guided Expectimax search (depth 3)
python scripts/evaluate.py data/models/NewArch-GradReward-v2/rl_model_24000000_steps.zip

# Evaluate using a deeper search for stronger play
python scripts/evaluate.py data/models/NewArch-GradReward-v2/rl_model_24000000_steps.zip --depth 5

# Evaluate using only the raw PPO policy (no search)
python scripts/evaluate.py data/models/NewArch-GradReward-v2/rl_model_24000000_steps.zip --no-search
```

## Project Structure

```
├── cpp_src/                # C++ source for the high-performance game core
├── data/
│   ├── models/             # Saved model checkpoints
│   └── studies.db          # Optuna hyperparameter tuning database
├── configs/                # YAML configuration files for training and tuning
├── scripts/                # Python scripts for training, evaluation, etc.
└── twenty_forty_eight_ai/  # Main Python package for the AI
    ├── agent/              # RL agent architecture and callbacks
    ├── env/                # Custom Gymnasium environment
    └── utils/              # Utility functions and the compiled C++ module
```
