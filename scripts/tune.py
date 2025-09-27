import optuna
import yaml
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from optuna_integration import WeightsAndBiasesCallback

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.architecture import CustomCNN

class TrialCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, report_freq: int = 100000, verbose=0):
        super(TrialCallback, self).__init__(verbose)
        self.trial = trial
        self.is_pruned = False
        self.report_freq = report_freq

    def _on_step(self) -> bool:
        # Check if it's time to report (e.g., every 10,000 steps)
        if self.num_timesteps % self.report_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                self.trial.report(mean_reward, self.num_timesteps)

                if self.trial.should_prune():
                    self.is_pruned = True
                    return False # Stop training
        return True

def _get_trial_suggester(trial: optuna.Trial, param_config: dict):
    """Helper function to parse the search space from the YAML config."""
    if param_config['type'] == 'categorical':
        return trial.suggest_categorical(param_config['name'], param_config['choices'])
    elif param_config['type'] == 'float':
        return trial.suggest_float(
            param_config['name'], param_config['low'], param_config['high'], log=param_config.get('log', False)
        )
    raise ValueError(f"Unsupported parameter type: {param_config['type']}")


def objective(trial: optuna.Trial, config: dict) -> float:
    # --- Suggest Hyperparameters from YAML Config ---
    ppo_params = {}
    for name, param_conf in config['ppo_search_space'].items():
        ppo_params[name] = _get_trial_suggester(trial, {'name': name, **param_conf})

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
        model.learn(total_timesteps=trial_config['total_timesteps'], callback=[pruning_callback])
    except AssertionError:
        raise optuna.exceptions.TrialPruned()

    if pruning_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    final_mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer if "r" in ep_info])
    return final_mean_reward if final_mean_reward is not np.nan else -1e9


if __name__ == '__main__':
    with open("configs/tune_config.yaml", "r") as f:
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
        # Pass the config dictionary to the objective function
        study.optimize(
            lambda trial: objective(trial, config),
            timeout=config['timeout_hours'] * 3600,
            show_progress_bar=False,
            callbacks=[wandb_callback]
        )
    except KeyboardInterrupt:
        print("\n--- OPTUNA STUDY INTERRUPTED ---")

    print("\n--- OPTUNA STUDY COMPLETE ---")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial:")
    trial = study.best_trial
    print(f"    Value: {trial.value}")
    print(f"    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")