class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("maskPPO_optuna_hyb_snake_run_1")

        self.CHECKPOINT_PATH = "models/maskPPO_optuna_mono_snake_run_1/rl_model_3000000_steps.zip"
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 5000000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 500000

        # Optuna hyperparameters
        self.CONFIG = {
            "n_steps": 2048,
            "gamma": 0.9798552085591075,
            "ent_coef": 0.0001532223928663341,
            "learning_rate": lambda progress_remaining: progress_remaining * 0.00021976112276314225,
            "clip_range": 0.2,

            "batch_size": 512,
            "n_epochs": 4,
        }
