import optuna
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from src.Game2048Env import Game2048Env
from src.PPO import CustomCNN


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


def objective(trial: optuna.Trial) -> float:
    hyperparams = {
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.05, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
    }

    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256))
    vec_env = make_vec_env(Game2048Env, n_envs=16)

    model = MaskablePPO(
        'CnnPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        **hyperparams,
        batch_size=512,
        n_epochs=4,
        verbose=0,
    )

    callback = TrialCallback(trial, report_freq=100000)

    try:
        model.learn(total_timesteps=2_000_000, callback=callback)
    except AssertionError:
        raise optuna.exceptions.TrialPruned()

    if callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    final_mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer if "r" in ep_info])
    return final_mean_reward if final_mean_reward is not np.nan else -1e9


if __name__ == '__main__':
    storage_name = "sqlite:///optuna_study.db"

    study_name = "2048-ppo-hynsnake-tuning"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=500_000)

    study = optuna.create_study(
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner
    )

    try:
        study.optimize(objective, n_trials=50, timeout=16 * 3600, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Study interrupted. Progress has been saved to the database.")

    print("\n--- OPTUNA STUDY COMPLETE ---")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial:")
    trial = study.best_trial
    print(f"    Value: {trial.value}")
    print(f"    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")