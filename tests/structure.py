from main import Connection, Model, Neuron
import numpy as np

def generate_layer(n: int):
    return [Neuron(np.random.uniform(-1,1)) for _ in range(n)]

def connect_layers(n: list[list[Neuron]]):
    for i, layer in enumerate(n):
        if i != 0:
            for neuron in layer:
                neuron.inputs = [Connection(np.random.uniform(-1,1), a_neuron) for a_neuron in n[i-1]]
    
    return n

def generate_neural_network(n_entrada: int, layers: list[int]):
    n = [generate_layer(n_entrada)]
    for l in layers:
        n.append(generate_layer(l))
    n.append(generate_layer(1))

    return connect_layers(n)

                

neural_net = generate_neural_network(1,[2])

model = Model(neural_net)

model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)
print("Modelo entrenado.")
print(model.predict([10]))
print(model.predict([35]))