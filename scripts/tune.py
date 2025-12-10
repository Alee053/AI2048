"""
Hyperparameter tuning with Optuna.
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
    """Callback for intermediate reporting and pruning."""
    def __init__(self, trial: optuna.Trial, report_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.is_pruned = False
        self.report_freq = report_freq

    def _on_step(self) -> bool:
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
    """Parse search space from config."""
    if param_config['type'] == 'categorical':
        return trial.suggest_categorical(name, param_config['choices'])
    elif param_config['type'] == 'float':
        return trial.suggest_float(
            name, param_config['low'], param_config['high'], log=param_config.get('log', False)
        )
    raise ValueError(f"Unsupported parameter type: {param_config['type']}")

def objective(trial: optuna.Trial, config: dict) -> float:
    """Optuna objective function."""
    # Suggest Hyperparameters
    ppo_params = {
        name: _get_trial_suggester(trial, name, param_conf)
        for name, param_conf in config['ppo_search_space'].items()
    }

    # Setup Environment
    trial_config = config['trial']
    vec_env = make_vec_env(Game2048Env, n_envs=trial_config['n_envs'])
    pruning_callback = TrialCallback(trial, report_freq=trial_config['report_freq'])

    # Train Model
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

    # Final value
    if not model.ep_info_buffer:
        return -1e9
    final_mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])
    return final_mean_reward

def main(config_path: str):
    """Run Optuna study."""
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