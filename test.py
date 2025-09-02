from src.Visualizer import Visualizer

vis=Visualizer()
vis.load_model("models/improved4.0_run_1/final_model.zip")
vis.test_agent()