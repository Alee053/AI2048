from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/new_ppo_run_1/final_model.zip")
vis.test_agent()