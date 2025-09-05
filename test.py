from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/maskPPO_run_2/rl_model_3000000_steps.zip")
vis.test_agent()