import os
import wandb
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from src.Config import Config
from src.Game2048Env import Game2048Env
from src.PPO import CustomCNN
from src.utility import CustomWandbCallback

conf = Config()

run = wandb.init(
    project=conf.PROJECT_NAME,
    config=conf.CONFIG,
    name=conf.RUN_NAME,
    save_code=True,
)
model_dir = f"models/{conf.RUN_NAME}/"
os.makedirs(model_dir, exist_ok=True)

vec_env = make_vec_env(Game2048Env, n_envs=conf.N_ENVS)

checkpoint_callback = CheckpointCallback(
    save_freq=max(conf.SAVE_INTERVAL // conf.N_ENVS, 1),
    save_path=model_dir,
    name_prefix="rl_model"
)
wandb_callback = CustomWandbCallback()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

should_load_model = conf.LOAD_MODEL and os.path.exists(conf.CHECKPOINT_PATH)

if should_load_model:
    print(f"Loading model from: {conf.CHECKPOINT_PATH}")
    model = MaskablePPO.load(conf.CHECKPOINT_PATH, env=vec_env, verbose=1)

    current_steps = model.num_timesteps
    remaining_steps = conf.TOTAL_TIMESTEPS - current_steps
    print(f"Model has been trained for {current_steps} steps.")

    if remaining_steps > 0:
        print(f"Training for an additional {remaining_steps} steps to reach {conf.TOTAL_TIMESTEPS}.")
        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, wandb_callback],
            progress_bar=True
        )
    else:
        print("Model has already been trained for the total number of timesteps. Exiting.")

else:
    if conf.LOAD_MODEL:
        print(f"Could not find model at {conf.CHECKPOINT_PATH}, training from scratch.")

    print("Starting a new training run.")
    model = MaskablePPO(
        conf.POLICY_TYPE,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **conf.CONFIG
    )
    model.learn(
        total_timesteps=conf.TOTAL_TIMESTEPS,
        reset_num_timesteps=True,
        callback=[checkpoint_callback, wandb_callback],
        progress_bar=True
    )

final_model_path = f"{model_dir}/final_model"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}.zip")
print("Training complete!")

run.finish()
