class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("maskPPO_optuna2.0_best_rew_CL_run_2")

        self.CHECKPOINT_PATH = ""
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 100_000_000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 1000000

        # Optuna hyperparameters
        self.CONFIG = {
            "n_steps": 512,
            "gamma": 0.9501603825891086,
            "ent_coef": 1.0029584256046895e-5,
            "learning_rate": lambda progress_remaining: progress_remaining * 0.0007726875334549466,
            "clip_range": 0.3,

            "batch_size": 512,
            "n_epochs": 4,
        }
