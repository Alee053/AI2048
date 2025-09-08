import optuna
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from src.Game2048Env import Game2048Env
from src.PPO import CustomCNN


# --- NEW: Callback for Pruning ---
class TrialCallback(BaseCallback):
    """
    Callback for reporting intermediate results and pruning trials.
    """

    def __init__(self, trial: optuna.Trial, verbose=0):
        super(TrialCallback, self).__init__(verbose)
        self.trial = trial
        self.is_pruned = False

    def _on_step(self) -> bool:
        # Report the mean reward of the last 100 episodes
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.trial.report(mean_reward, self.num_timesteps)

            # Prune trial if it is unpromising
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Stop training
        return True


def objective(trial: optuna.Trial) -> float:
    # Your hyperparameter suggestions remain the same
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
        verbose=1,  # Set to 0 to keep logs clean
    )

    # --- NEW: Use the callback ---
    callback = TrialCallback(trial)

    try:
        # Train for the full 2M steps
        model.learn(total_timesteps=2_000_000, callback=callback)
    except AssertionError as e:
        # Sometimes SB3 throws an error on early termination. Catch it.
        if callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    # If the trial was pruned, Optuna handles it automatically
    if callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # Get the final mean reward for the completed trial
    final_mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])
    return final_mean_reward


# --- Start the Optimization ---
if __name__ == '__main__':
    # --- NEW: Add a pruner to the study ---
    # The MedianPruner stops trials that are performing worse than the median.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=500_000)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Run for 30 trials. Optuna will manage the pruning.
    study.optimize(objective, n_trials=30, timeout=12 * 3600, show_progress_bar=True)

    # Save the results (same as before)
    with open('hyperparameters/optuna_output_2.txt', 'w') as f:
        f.write("Study statistics: \n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")
        trial = study.best_trial
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")