class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = "improved6.0_run_1"

        self.CHECKPOINT_PATH = "models/improved3.0_run_2_finetune2.0/final_model.zip"
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 50000000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 500000

        self.CONFIG = {
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "ent_coef": 0.02,
            "learning_rate": lambda progress_remaining: progress_remaining * 5e-5,
            "clip_range": 0.2,
        }
