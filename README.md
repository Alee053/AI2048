# 2048 AI Agent

This repository contains the code for a deep reinforcement learning agent designed to play the game 2048. The agent uses a custom "2048-Aware" CNN architecture and is trained using Proximal Policy Optimization (PPO). For enhanced performance during evaluation, the trained agent can be combined with a C++-based Expectimax searcher.

## Key Features

- **Custom CNN Architecture**: The model uses a unique neural network architecture that processes the board in three parallel pathways: by row, by column, and by 2x2 grid. This helps the agent recognize spatial patterns specific to 2048.
- **High-Performance Game Engine**: The core 2048 game logic is implemented in C++ and uses look-up tables for maximum speed, allowing for faster training and deeper search.
- **PPO + Expectimax Search**: While the base agent is trained with PPO, it can be augmented at evaluation time with an Expectimax search algorithm. This search uses the agent's learned value function (critic) to explore the game tree and make more robust tactical decisions.
- **Configuration-Driven**: Training and hyperparameter tuning are managed through simple YAML configuration files, making it easy to experiment with different settings.
- **W&B Integration**: The training and tuning scripts are fully integrated with Weights & Biases for seamless experiment tracking, logging, and visualization.

## Project Structure

```
.
├── configs/                # YAML configuration files for training and tuning
├── cpp_src/                # C++ source for the high-performance game engine and searcher
├── data/                   # Directory for storing models and other data
├── scripts/                # Python scripts for training, evaluation, and tuning
└── twenty_forty_eight_ai/  # Main Python source code for the project
    ├── agent/              # RL agent components (architecture, callbacks)
    ├── env/                # Gymnasium environment for the 2048 game
    └── utils/              # Utility functions (visualizer, tensor manipulation)
```

## Setup and Installation

### Prerequisites

- C++ Compiler (with C++17 support, e.g., GCC, Clang, MSVC)
- Python 3.9 or higher
- [Poetry](httpss://python-poetry.org/docs/#installation) for managing Python dependencies

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/2048-ai.git
    cd 2048-ai
    ```

2.  **Install Python dependencies using Poetry:**
    This command will create a virtual environment and install all necessary Python packages listed in `pyproject.toml`.
    ```bash
    poetry install
    ```

3.  **Compile the C++ components:**
    This step builds the C++ game engine and Expectimax searcher and creates Python bindings, making them available to the Python code.
    ```bash
    poetry run python setup.py install
    ```
    This command invokes the `setup.py` script, which uses `scikit-build` to handle the CMake build process and install the compiled module into the Poetry environment.

## Usage

### Training a New Agent

1.  **Configure your run**: Modify the `configs/train_config.yaml` file to set your desired parameters. You can specify the number of training timesteps, learning rate, environment settings, and more.
2.  **Start training**: Run the training script with your configuration file.
    ```bash
    poetry run python scripts/train.py --config configs/train_config.yaml
    ```
    Progress will be logged to the console and to Weights & Biases. Model checkpoints will be saved periodically in the `data/models/` directory.

### Evaluating a Trained Agent

You can watch a trained agent play the game using the Pygame-based visualizer.

-   **To run with PPO + Expectimax search (recommended for best performance):**
    ```bash
    poetry run python scripts/evaluate.py data/models/<your_run_name>/final_model.zip
    ```
-   **To run with only the raw PPO policy (no search):**
    ```bash
    poetry run python scripts/evaluate.py data/models/<your_run_name>/final_model.zip --no-search
    ```

### Tuning Hyperparameters

The project uses Optuna for hyperparameter optimization.

1.  **Define a search space**: Create or modify a YAML file in the `configs/tune/` directory. This file defines the study name, database location, and the hyperparameter search space.
2.  **Run the study**:
    ```bash
    poetry run python scripts/tune.py --config configs/tune/my_study.yaml
    ```
    Optuna will run multiple training trials to find the best combination of hyperparameters. Results will be saved to the specified SQLite database and logged to Weights & Biases.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.