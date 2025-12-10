"""
Training script for 2048 AI.
"""

import os
import yaml
import argparse
import wandb
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.agent.callbacks import WandbLoggingCallback


def train(config: dict):
    """Run training loop."""
    run = wandb.init(
        project=config['project_name'],
        config=config,
        name=config['run_name'],
        save_code=True,
    )

    model_dir = os.path.join(config['output_dir'], "models", config['run_name'])
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks
    wandb_callback = WandbLoggingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config['save_interval'] // config['n_envs'], 1),
        save_path=model_dir,
        name_prefix="rl_model"
    )
    callbacks = [wandb_callback, checkpoint_callback]

    # Environment & Model
    vec_env = make_vec_env(Game2048Env, n_envs=config['n_envs'])
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config['features_dim']),
    )

    should_load_model = config['load_model'] and config['checkpoint_path'] is not None

    if should_load_model:
        print(f"Loading model from: {config['checkpoint_path']}")
        if not os.path.exists(config['checkpoint_path']):
            print(f"Error: Checkpoint path not found at {config['checkpoint_path']}. Exiting.")
            return

        model = MaskablePPO.load(config['checkpoint_path'], env=vec_env, verbose=1)

        # Calculate remaining steps
        current_steps = model.num_timesteps
        total_steps = config['total_timesteps']
        remaining_steps = total_steps - current_steps

        # Resumed LR schedule
        lr_config = config['ppo_params']['learning_rate']
        if lr_config['type'] == 'linear_decay':
            def resumed_lr_schedule(progress_remaining: float) -> float:
                progress_this_run = 1.0 - progress_remaining
                steps_this_run = progress_this_run * remaining_steps
                current_global_steps = current_steps + steps_this_run
                global_progress_remaining = 1.0 - (current_global_steps / total_steps)
                return global_progress_remaining * lr_config['initial_value']

            model.learning_rate = resumed_lr_schedule

        print(f"Resuming training for an additional {remaining_steps} steps.")
        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=False
        )

    else:
        print("Starting a new training run from scratch.")
        ppo_params = config['ppo_params'].copy()

        # New LR schedule
        lr_config = ppo_params.pop('learning_rate')
        if lr_config['type'] == 'linear_decay':
            ppo_params['learning_rate'] = lambda p: p * lr_config['initial_value']

        model = MaskablePPO(
            "CnnPolicy", vec_env, policy_kwargs=policy_kwargs,
            verbose=1, **ppo_params
        )
        model.learn(
            total_timesteps=config['total_timesteps'],
            reset_num_timesteps=True,
            callback=callbacks,
            progress_bar=False
        )

    # Save model
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    try:
        train(config_data)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
    finally:
        if wandb.run is not None:
            wandb.finish()