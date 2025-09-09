from src.Visualizer import Visualizer

MODEL_PATH = "models/maskPPO_optuna2.0_newmono_run_1_finetune_100M/final_model.zip"
USE_EXPECTIMAX = True


if __name__ == '__main__':
    try:
        vis = Visualizer(model_path=MODEL_PATH,use_expectimax=USE_EXPECTIMAX)
        vis.run_visualization()
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
