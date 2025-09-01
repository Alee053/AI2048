class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = "ppo_run_5_finetune"

        self.CHECKPOINT_PATH = "models/ppo_run_4_finetune/final_model.zip"
        self.LOAD_MODEL = True

        self.TOTAL_TIMESTEPS = 10000000
        self.POLICY_TYPE = "CnnPolicy"

        self.CONFIG = {
            "n_envs": 16,
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "ent_coef": 0.015,
            "learning_rate": 5e-5,
            "clip_range": 0.1,
        }
