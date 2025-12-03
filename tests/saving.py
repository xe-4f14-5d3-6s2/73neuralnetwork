import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Connection, Model, Neuron

model = Model.new(1, [])
model.save()