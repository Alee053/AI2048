from Trainer import Trainer


model=Trainer("model_5")

#model.load_model("model_5")
model.train_model(10000,1000)
model.test_model()
