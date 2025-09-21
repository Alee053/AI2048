import os
import wandb
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from src.Config import Config
from src.Game2048Env import Game2048Env
from src.PPO import CustomCNN
from src.utility import AdaptiveCurriculumCheckpointCallback,WandbLoggingCallback
from collections import deque

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


wandb_callback = WandbLoggingCallback()
acl_checkpoint_callback= AdaptiveCurriculumCheckpointCallback(save_freq=max(conf.SAVE_INTERVAL // conf.N_ENVS, 1),
                                                              save_path=model_dir,
                                                              name_prefix="rl_model")

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

    # --- THE FIX: Look for the attribute directly on the loaded model ---
    if hasattr(model, 'curriculum_state'):
        state_data = model.curriculum_state
        # Restore the state of our new callback instance
        acl_checkpoint_callback.difficulty_level = state_data["difficulty_level"]
        acl_checkpoint_callback.reward_threshold = state_data["reward_threshold"]
        acl_checkpoint_callback.reward_buffer = deque(state_data["reward_buffer"], maxlen=100)
        acl_checkpoint_callback.is_initialized = state_data["is_initialized"]
        print("Resumed curriculum state successfully.")
    else:
        print("WARNING: Could not find curriculum state in the loaded model. Starting curriculum from scratch.")

    model.learn(
        total_timesteps=remaining_steps,
        reset_num_timesteps=False,
        callback=[wandb_callback, acl_checkpoint_callback],
        progress_bar=True
    )

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
        callback=[wandb_callback, acl_checkpoint_callback],
        progress_bar=True
    )

final_model_path = f"{model_dir}/final_model"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}.zip")
print("Training complete!")

run.finish()
