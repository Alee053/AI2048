class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("ACL_2.0_new_arch_run_1")

        self.CHECKPOINT_PATH = "models/ACL_2.0_run_1/rl_model_49500000_steps.zip"
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 50_000_000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 1000000

        self.INITIAL_LR = 0.0007726875334549466

        # Optuna hyperparameters
        self.CONFIG = {
            "n_steps": 512,
            "gamma": 0.9501603825891086,
            "ent_coef": 1.0029584256046895e-5,
            "learning_rate": lambda progress_remaining: progress_remaining * self.INITIAL_LR,
            "clip_range": 0.3,

            "batch_size": 512,
            "n_epochs": 4,
        }
