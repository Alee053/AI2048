class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = "new_ppo_run_2_finetune"

        self.CHECKPOINT_PATH = "../models/new_ppo_run_1/final_model.zip"
        self.LOAD_MODEL = True

        self.TOTAL_TIMESTEPS = 5000000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 500000

        self.CONFIG = {
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "ent_coef": 0.015,
            "learning_rate": 5e-5,
            "clip_range": 0.1,
        }
