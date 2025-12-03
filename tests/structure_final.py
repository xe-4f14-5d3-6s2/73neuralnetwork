from ..main import Connection, Model, Neuron

model = Model.new(1, [])

model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)
print("Modelo entrenado.")
print(model.predict([10]))
print(model.predict([35]))