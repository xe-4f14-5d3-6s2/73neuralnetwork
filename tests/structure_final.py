import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Connection, Model, Neuron

print("Creando Modelo...")
model = Model.new(1, [])
print("Modelo Creado.")
print("-"*20)
print("Entrenando Modelo...")
model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)
print("Modelo entrenado.")
print(model.predict([10]))
print(model.predict([35]))