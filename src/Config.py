class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = "maskPPO_run_4"

        self.CHECKPOINT_PATH = "models/maskPPO_run_3/rl_model_4200000_steps.zip"
        self.LOAD_MODEL = True

        self.TOTAL_TIMESTEPS = 50000000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 300000

        self.CONFIG = {
            "n_steps": 1024,
            "batch_size": 512,
            "n_epochs": 4,
            "gamma": 0.99,
            "ent_coef": 0.02,
            "learning_rate": lambda progress_remaining: progress_remaining * 3e-4,
            "clip_range": 0.2,
        }
