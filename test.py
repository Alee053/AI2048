from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/improved3.0_run_1/rl_model_3300000_steps.zip")
vis.test_agent()