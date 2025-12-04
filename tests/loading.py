import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Connection, Model, Neuron

model = Model.load("/home/tantalumdev/Documents/Proyectos/RedNeuronal/56cf25bc-2f47-427d-9895-1af339846de0.73nn")

model.predict([35])