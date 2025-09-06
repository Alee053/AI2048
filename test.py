from src.Visualizer import Visualizer

MODEL_PATH = "models/maskPPO_optuna_run_1_continue2.0/rl_model_23100000_steps.zip"


if __name__ == '__main__':
    try:
        vis = Visualizer(model_path=MODEL_PATH)
        vis.run_visualization()
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
