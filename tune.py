import optuna
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from src.Game2048Env import Game2048Env
from src.PPO import CustomCNN


def objective(trial: optuna.Trial) -> float:
    hyperparams = {
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
    }

    policy_kwargs = dict(features_extractor_class=CustomCNN,features_extractor_kwargs=dict(features_dim=256))
    vec_env = make_vec_env(Game2048Env, n_envs=16)

    model = MaskablePPO(
        'CnnPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        **hyperparams,
        batch_size=512,
        n_epochs=4,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)

    mean_reward = model.ep_info_buffer[-1]['r']

    return mean_reward

# --- Start the Optimization ---
if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=8*3600,show_progress_bar=True)

    with open('hyperparameters/optuna_output_1.txt', 'w') as f:
        f.write("Study statistics: \n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")
        trial = study.best_trial
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")