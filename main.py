from Model import Model


model=Model("model_4")

#model.load_model("model_2")
model.train_model(5000,500)
model.test_model()
