import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Connection, Model, Neuron

model = Model.new(1, [])

model.train([([c], ((c*(9/5))+32)) for c in range(500)], 100, 0.000001, 0.001)

model.save()