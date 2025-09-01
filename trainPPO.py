import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import wandb


from architecture.PPO import CustomCNN
from game.Game2048Env import Game2048Env


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]

                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']

                    wandb.log({
                        "Episode Reward Mean": episode_reward,
                        "Episode Length Mean": episode_length,
                    })

                wandb.log({"Score": info['score']})
                wandb.log({"Max Tile": 2**info['max_tile']})

        return True



config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 20_000_000,
    "n_envs": 16,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "ent_coef": 0.02,
    "learning_rate": 2.5e-4,
    "clip_range": 0.2,
}
RUN_NAME = "ppo_run_2_finetune"

BEST_CHECKPOINT_PATH = "models/ppo_run_1/final_model.zip"
LOAD_MODEL = True



run = wandb.init(
    project="2048-ppo",
    config=config,
    name=RUN_NAME,
    sync_tensorboard=True,
    save_code=True,
)
model_dir = f"models/{RUN_NAME}/"
os.makedirs(model_dir, exist_ok=True)

vec_env = make_vec_env(Game2048Env, n_envs=config["n_envs"])

checkpoint_callback = CheckpointCallback(
    save_freq=max(100_000 // config["n_envs"], 1),
    save_path=model_dir,
    name_prefix="rl_model"
)

wandb_callback = CustomWandbCallback()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),  # Output 256 features
)

if LOAD_MODEL:
    model = PPO.load(BEST_CHECKPOINT_PATH, env=vec_env)
else:
    model = PPO(
        'CnnPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
    )

model.learn(total_timesteps=config["total_timesteps"],reset_num_timesteps=not LOAD_MODEL, callback=[checkpoint_callback,wandb_callback], progress_bar=True)

run.finish()


model.save(f"{model_dir}/final_model")

print(f"Final model saved to: {model_dir}/final_model.zip")
print("Training complete!")
