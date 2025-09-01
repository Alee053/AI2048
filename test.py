from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/ppo_run_5_finetune/final_model.zip")
vis.test_agent()