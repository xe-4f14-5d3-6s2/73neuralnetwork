import numpy as np 


class Neurona:
    def __init__(self, bias: float = 0):
        self.entradas:list[Enlace] = []
        self.bias = bias
        
        self.value: float = 0
    
    def __str__(self):
        return f"| {self.value} |"
    
    def __repr__(self):
        return f"| {self.value} |"

class Enlace:
    def __init__(self, peso: float, neurona: Neurona):
        self.peso = peso
        self.neurona = neurona        
    
class Model:
    def __init__(self, red_neuronal: list[list[Neurona]], compression:str = "ReLU"):
        self.red_neuronal = red_neuronal
        self.compression = compression
        
    def compress(self, n: float):
        if self.compression == "ReLU":
            return 0 if n <= 0 else n
        if self.compression == "LeakyReLU":
            return n if n >= 0 else 0.01 * n
        elif self.compression == "Sigmoid":
            return 1 / (1 + np.exp(-n))
        elif self.compression == "Binary":
            return 0 if n <= 0 else 1
        elif self.compression == "Sign":
            return -1 if n <= 0 else 1
        else:
            print("Función de compresión desconocida.")
            return 0
        
    def activation(self, entradas: list[Enlace], bias: float):
        W = np.array([e.peso for e in entradas])
        A = np.array([e.neurona.value for e in entradas])
        
        z = np.sum(W * A) + bias
        
        return self.compress(z)
     
    def train(self, dataset: list[tuple[list[float], float]], epochs=500, lr_w=0.001, lr_b=0.01):
        for _ in range(epochs):
            for entrada, salida_real in dataset:
                salida_pred = self.predict(entrada)
                 
                e = salida_real - salida_pred.value
                
                for capa in self.red_neuronal[1:]:
                    for neurona in capa:
                        for enlace in neurona.entradas:
                            enlace.peso += lr_w * e * enlace.neurona.value
                        neurona.bias += lr_b * e
        
    def predict(self, data: list[float]):
        if len(data) != len(self.red_neuronal[0]): 
            return "Los datos no coinciden con los parámetros de entrada."
         
        for i, capa in enumerate(self.red_neuronal):
            if i == 0:
                for i, neurona in enumerate(capa):
                    neurona.value = data[i]
            else:
                for i, neurona in enumerate(capa):
                    neurona.value = self.activation(neurona.entradas, neurona.bias)
        
        
        return self.red_neuronal[-1][0]


n1 = Neurona()
n2 = Neurona(np.random.uniform(-1,1))

n2.entradas = [
    Enlace(np.random.uniform(-1,1), n1)
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