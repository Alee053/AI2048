from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/improved3.0_run_3_finetune2.0/final_model.zip")
vis.test_agent()