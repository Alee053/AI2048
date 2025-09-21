class Config:
    def __init__(self):
        self.PROJECT_NAME = "2048-ppo"

        self.RUN_NAME = ("maskPPO_optuna5.0_best_rew_new_arch_run_1")

        self.CHECKPOINT_PATH = "models/maskPPO_optuna2.0_best_rew_ACL_run_1/rl_model_31000000_steps.zip"
        self.LOAD_MODEL = False

        self.TOTAL_TIMESTEPS = 200_000_000
        self.POLICY_TYPE = "CnnPolicy"

        self.N_ENVS = 16

        self.SAVE_INTERVAL = 1000000

        self.INITIAL_LR = 0.0006424222145120949

        # Optuna hyperparameters
        self.CONFIG = {
            "n_steps": 512,
            "gamma": 0.9628863708245955,
            "ent_coef": 0.00021322091058994733,
            "learning_rate": lambda progress_remaining: progress_remaining * self.INITIAL_LR,
            "clip_range": 0.3,

            "batch_size": 512,
            "n_epochs": 4,
        }
