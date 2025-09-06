class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("maskPPO_optuna_run_1_continue3.0")

        self.CHECKPOINT_PATH = "models/maskPPO_optuna_run_1_continue2.0/rl_model_23100000_steps.zip"
        self.LOAD_MODEL = True

        self.TOTAL_TIMESTEPS = 100000000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 300000

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
