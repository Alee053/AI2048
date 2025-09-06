from src.Visualizer import Visualizer

MODEL_PATH = "models/maskPPO_optuna_run_1_continue/rl_model_20700000_steps.zip"


if __name__ == '__main__':
    try:
        vis = Visualizer(model_path=MODEL_PATH)
        vis.run_visualization()
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
