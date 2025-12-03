import numpy as np 


class Neuron:
    def __init__(self, bias: float = 0):
        self.inputs:list[Connection] = []
        self.bias = bias
        
        self.value: float = 0
    
    def __str__(self):
        return f"| {self.value} |"
    
    def __repr__(self):
        return f"| {self.value} |"

class Connection:
    def __init__(self, weight: float, neuron: Neuron):
        self.weight = weight
        self.neuron = neuron        
    
class Model:
    def __init__(self, neural_network: list[list[Neuron]], activation_function:str = "ReLU"):
        self.neural_network = neural_network
        self.activation_function = activation_function
        
    def compress(self, n: float):
        if self.activation_function == "ReLU":
            return 0 if n <= 0 else n
        if self.activation_function == "LeakyReLU":
            return n if n >= 0 else 0.01 * n
        elif self.activation_function == "Sigmoid":
            return 1 / (1 + np.exp(-n))
        elif self.activation_function == "Binary":
            return 0 if n <= 0 else 1
        elif self.activation_function == "Sign":
            return -1 if n <= 0 else 1
        else:
            print("Función de compresión desconocida.")
            return 0
        
    def activate(self, inputs: list[Connection], bias: float):
        W = np.array([e.weight for e in inputs])
        A = np.array([e.neuron.value for e in inputs])
        
        z = np.sum(W * A) + bias
        
        return self.compress(z)
     
    def train(self, dataset: list[tuple[list[float], float]], epochs=500, learning_rate_w=0.001, learning_rate_b=0.01):
        for _ in range(epochs):
            for inp, expected_output in dataset:
                predicted_ouput = self.predict(inp)
                 
                e = expected_output - predicted_ouput.value
                
                for layer in self.neural_network[1:]:
                    for neuron in layer:
                        for enlace in neuron.inputs:
                            enlace.weight += learning_rate_w * e * enlace.neuron.value
                        neuron.bias += learning_rate_b * e
        
    def predict(self, data: list[float]):
        if len(data) != len(self.neural_network[0]): 
            return "Los datos no coinciden con los parámetros de inp."
         
        for i, layer in enumerate(self.neural_network):
            if i == 0:
                for i, neuron in enumerate(layer):
                    neuron.value = data[i]
            else:
                for i, neuron in enumerate(layer):
                    neuron.value = self.activate(neuron.inputs, neuron.bias)
        
        
        return self.neural_network[-1][0]


n1 = Neuron()
n2 = Neuron(np.random.uniform(-1,1))

n2.inputs = [
    Connection(np.random.uniform(-1,1), n1)
]

model = Model([
    [n1],
    [n2],
], "LeakyReLU")
# 
model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)
print("Modelo entrenado.")
print(model.predict([10]))
print(model.predict([35]))