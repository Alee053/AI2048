"""
Main hyperparameter tuning script for the 2048 AI agent using Optuna.

This script is configuration-driven. It loads a YAML file that defines the
study parameters, pruner settings, and the hyperparameter search space.

Usage:
    python scripts/tune.py --config configs/tune/NewArch-GradReward-v1.yaml
"""

import argparse
import yaml
import optuna
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from optuna_integration import WeightsAndBiasesCallback

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.architecture import CustomCNN

class TrialCallback(BaseCallback):
    """A custom callback for Optuna to report intermediate values and handle pruning.

    This callback is used within a Stable Baselines 3 training loop. It
    periodically reports the mean reward to the Optuna trial. Based on this
    reported value, Optuna's pruner can decide to stop an unpromising trial
    early.

    Attributes:
        trial (optuna.Trial): The Optuna trial associated with this callback.
        is_pruned (bool): A flag that is set to True if the trial should be pruned.
        report_freq (int): The frequency (in timesteps) at which to report
            the mean reward to the trial.
    """
    def __init__(self, trial: optuna.Trial, report_freq: int, verbose: int = 0):
        """Initializes the TrialCallback.

        Args:
            trial (optuna.Trial): The Optuna trial to report to.
            report_freq (int): The frequency for reporting intermediate values.
            verbose (int, optional): The verbosity level. Defaults to 0.
        """
        super().__init__(verbose)
        self.trial = trial
        self.is_pruned = False
        self.report_freq = report_freq

    def _on_step(self) -> bool:
        """Called by Stable Baselines 3 after each step.

        Checks if it's time to report the mean reward. If so, it reports to
        the trial and checks if the trial should be pruned.

        Returns:
            bool: False if the training should be stopped (i.e., pruned),
            True otherwise.
        """
        if self.num_timesteps % self.report_freq == 0:
            if self.model.ep_info_buffer:
                # Use the mean reward from the episode buffer for reporting
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                self.trial.report(mean_reward, self.num_timesteps)
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False  # Stop training
        return True

def _get_trial_suggester(trial: optuna.Trial, name: str, param_config: dict):
    """Parses the search space from the YAML config and suggests a value.

    This helper function reads a parameter's configuration (type, range, etc.)
    from the dictionary loaded from the YAML file and calls the appropriate
    `trial.suggest_*` method from Optuna.

    Args:
        trial (optuna.Trial): The current Optuna trial.
        name (str): The name of the hyperparameter.
        param_config (dict): The configuration for this parameter.

    Raises:
        ValueError: If an unsupported parameter type is specified in the config.

    Returns:
        The suggested value for the hyperparameter for the current trial.
    """
    if param_config['type'] == 'categorical':
        return trial.suggest_categorical(name, param_config['choices'])
    elif param_config['type'] == 'float':
        return trial.suggest_float(
            name, param_config['low'], param_config['high'], log=param_config.get('log', False)
        )
    raise ValueError(f"Unsupported parameter type: {param_config['type']}")

def objective(trial: optuna.Trial, config: dict) -> float:
    """The main objective function that Optuna will optimize.

    Each call to this function constitutes one "trial" in the Optuna study.
    It performs the following steps:
    1.  Suggests a set of hyperparameters based on the search space defined
        in the configuration file.
    2.  Sets up the environment and a `TrialCallback` for pruning.
    3.  Creates and trains a MaskablePPO model with the suggested hyperparameters.
    4.  Returns the final mean reward achieved by the model, which Optuna
        uses as the objective value to maximize.

    Args:
        trial (optuna.Trial): The Optuna trial object, used for suggesting
            parameters and reporting results.
        config (dict): The full configuration dictionary loaded from the YAML file.

    Returns:
        float: The final mean reward of the trained model, which is the
        objective value for this trial.
    """
    # --- Suggest Hyperparameters from YAML Config ---
    ppo_params = {
        name: _get_trial_suggester(trial, name, param_conf)
        for name, param_conf in config['ppo_search_space'].items()
    }

    # --- Setup Environment and Callbacks ---
    trial_config = config['trial']
    vec_env = make_vec_env(Game2048Env, n_envs=trial_config['n_envs'])
    pruning_callback = TrialCallback(trial, report_freq=trial_config['report_freq'])

    # --- Create and Train Model ---
    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256))
    model = MaskablePPO(
        'CnnPolicy', vec_env, policy_kwargs=policy_kwargs,
        **ppo_params, batch_size=512, n_epochs=4, verbose=0
    )

    try:
        model.learn(total_timesteps=trial_config['total_timesteps'], callback=pruning_callback)
    except AssertionError:
        raise optuna.exceptions.TrialPruned()

    if pruning_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # Use the final mean reward from the buffer as the trial's value
    if not model.ep_info_buffer:
        return -1e9 # Return a very low value if no episodes finished
    final_mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])
    return final_mean_reward

def main(config_path: str):
    """Loads configuration, sets up, and runs the Optuna study."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pruner_config = config['pruner']

    storage = f"sqlite:///{config['db_path']}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_config['n_startup_trials'],
        n_warmup_steps=pruner_config['n_warmup_steps']
    )

    wandb_kwargs = {
        "project": config['project_name'],
        "group": config['study_name']
    }
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="final_mean_reward",
        wandb_kwargs=wandb_kwargs
    )

    study = optuna.create_study(
        storage=storage, study_name=config['study_name'],
        load_if_exists=True, direction="maximize", pruner=pruner
    )

    try:
        study.optimize(
            lambda trial: objective(trial, config),
            timeout=config['timeout_hours'] * 3600,
            callbacks=[wandb_callback],
            show_progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n--- OPTUNA STUDY INTERRUPTED BY USER ---")

    print("\n--- OPTUNA STUDY COMPLETE ---")
    if study.best_trial:
        print(f"Best trial for study '{study.study_name}':")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an Optuna hyperparameter study for the 2048 AI.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file for the tuning study."
    )
    args = parser.parse_args()
    main(config_path=args.config)