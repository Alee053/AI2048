class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("maskPPO_optuna2.0_newmono_run_1")

        self.CHECKPOINT_PATH = "models/maskPPO_optuna_mono_snake_run_1/rl_model_3000000_steps.zip"
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 50000000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 500000

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
