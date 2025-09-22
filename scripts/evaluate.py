from src.Visualizer import Visualizer

MODEL_PATH = "../data/models/pre-refactor/ACL_2.0_run_1_continue/rl_model_71500000_steps.zip"
USE_EXPECTIMAX = True


if __name__ == '__main__':
    try:
        vis = Visualizer(model_path=MODEL_PATH,use_expectimax=USE_EXPECTIMAX)
        vis.run_visualization()
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
