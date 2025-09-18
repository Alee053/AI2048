class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("maskPPO_optuna3.0_snake_run_1_100M")

        self.CHECKPOINT_PATH = "models/maskPPO_optuna2.0_newmono_run_1_continue/final_model.zip"
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 100_000_000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 1000000

        # Optuna hyperparameters
        self.CONFIG = {
            "n_steps": 1024,
            "gamma": 0.9830543714751623,
            "ent_coef": 0.00023322484726855512,
            "learning_rate": lambda progress_remaining: progress_remaining * 0.00005435281101357809,
            "clip_range": 0.3,

            "batch_size": 512,
            "n_epochs": 4,
        }
