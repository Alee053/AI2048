import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import wandb
from src.Game2048Env import Game2048Env
from src.PPO import CustomCNN
from src.Config import Config
from src.utility import CustomWandbCallback

# Load configuration
conf=Config()

run = wandb.init(
    project=conf.PROJECT_NAME,
    config=conf.CONFIG,
    name=conf.RUN_NAME,
    save_code=True,
)
model_dir = f"models/{conf.RUN_NAME}/"
os.makedirs(model_dir, exist_ok=True)

vec_env = make_vec_env(Game2048Env, n_envs=conf.N_ENVS)

# Callbacks
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

load=conf.LOAD_MODEL and os.path.exists(conf.CHECKPOINT_PATH)

if load:
    model = MaskablePPO.load(conf.CHECKPOINT_PATH, env=vec_env)
else:
    if conf.LOAD_MODEL:
        print(f"Couldnt find the model at {conf.CHECKPOINT_PATH}, training from scratch")
    model = MaskablePPO(
        conf.POLICY_TYPE,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **conf.CONFIG
    )

model.learn(total_timesteps=conf.TOTAL_TIMESTEPS,reset_num_timesteps=not load, callback=[checkpoint_callback,wandb_callback], progress_bar=True)

run.finish()


model.save(f"{model_dir}/final_model")

print(f"Final model saved to: {model_dir}/final_model.zip")
print("Training complete!")
