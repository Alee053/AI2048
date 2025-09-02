from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/improved_run_2/final_model.zip")
vis.test_agent()