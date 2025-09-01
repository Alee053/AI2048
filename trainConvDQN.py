from utility.ConvDQNTrainer import Trainer


model=Trainer("150k_ConvDQN_model")

#model.load_model("model_7")
model.train_model(1000000,5000)
model.test_model()
