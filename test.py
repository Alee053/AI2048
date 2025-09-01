from Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/ppo_run_2_finetune/final_model.zip")
vis.test_agent()