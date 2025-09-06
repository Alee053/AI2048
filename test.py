from src.Visualizer import Visualizer

MODEL_PATH = "models/maskPPO_optuna_mono_snake_run_2/final_model.zip"
USE_EXPECTIMAX = False


if __name__ == '__main__':
    try:
        vis = Visualizer(model_path=MODEL_PATH,use_expectimax=USE_EXPECTIMAX)
        vis.run_visualization()
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
