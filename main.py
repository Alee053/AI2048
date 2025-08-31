from Trainer import Trainer


model=Trainer("1M_episode_model")

#model.load_model("model_7")
model.train_model(1000000,5000)
model.test_model()
