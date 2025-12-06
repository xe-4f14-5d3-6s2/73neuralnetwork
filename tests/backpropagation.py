import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Model

model = Model.new(2,1,[4,4],"LeakyReLU")

masses = list(range(50))
accels = list(range(50))

data = []
for m in masses:
    for a in accels:
        data.append((m,a))

max_mass = max(masses)
max_accel = max(accels)
max_force = max_mass * max_accel

dataset = [
    ([m/max_mass, a/max_accel], [(m*a)/max_force])
    for m,a in data
]

epochs = 10
learning_rate_w = 0.0001
learning_rate_b = 0.0001

model.train(dataset, epochs, learning_rate_w, learning_rate_b)

test_mass = 2
test_accel = 2

norm_input = [test_mass/max_mass, test_accel/max_accel]
output = model.predict(norm_input)

pred_force = output[0].a * max_force

print(f"m={test_mass}, a={test_accel}, F_pred={pred_force}, F_real={test_mass*test_accel}")
