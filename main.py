import numpy as np
import logging, uuid, zlib, json, os

DEBUG = False

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO, format="[%(levelname)s] %(message)s"
)


class Neuron:
    def __init__(self, bias: float = 0):
        self.inputs: list[Connection] = []
        self.bias = bias
        self.a: float = 0
        self.z: float = 0

    def __repr__(self):
        return f"({self.a})"


class Connection:
    def __init__(self, weight: float, neuron: Neuron):
        self.weight = weight
        self.neuron = neuron


class Model:
    def __init__(self):
        self.neural_network: list[list[Neuron]] = []
        self.activation_function = ""

    @staticmethod
    def validate_file(data: list):
        if not isinstance(data, list):
            return False

        for layer in data:
            if not isinstance(layer, list):
                return False

            for neuron in layer:
                if not isinstance(neuron, list) or len(neuron) != 2:
                    return False

                bias, weights = neuron

                if not isinstance(bias, (int, float)):
                    return False

                if not isinstance(weights, list):
                    return False

                for weight in weights:
                    if not isinstance(weight, (int, float)):
                        return False
        return True

    @staticmethod
    def load(path: str):
        path = os.path.abspath(path)

        if not os.path.isfile(path):
            logging.info("Error: El archivo no existe.")
            return 0

        with open(path, "rb") as file_model:
            model_data = zlib.decompress(file_model.read())

        model_data = json.loads(model_data)
        if not Model.validate_file(model_data):
            logging.info("Error: El archivo no es válido o está corrupto.")
            return 0

        first_layer_neurons = len(model_data[0])
        hidden_layers_neurons = [len(layer) for layer in model_data[1:]] if len(model_data) >= 3 else []

        model = Model.new(first_layer_neurons, len(model_data[-1]), hidden_layers_neurons)

        for i_l, layer in enumerate(model.neural_network[1:]):
            for i_n, neuron in enumerate(layer):
                neuron.bias = model_data[i_l + 1][i_n][0]
                for i_c, connection in enumerate(neuron.inputs):
                    connection.weight = model_data[i_l + 1][i_n][1][i_c]

        logging.info("Modelo cargado y configurado exitosamente.")
        return model

    def save(self):
        model_data = [
            [[n.bias, [e.weight for e in n.inputs]] for n in layer]
            for layer in self.neural_network
        ]

        name = f"{uuid.uuid4()}.73nn"
        data_string = f"{json.dumps(model_data)}".encode("utf-8")

        try:
            with open(name, "wb") as f:
                f.write(zlib.compress(data_string, 9))
                logging.info(f"Modelo guardado exitosamente como {name}")
                return 1
        except Exception as e:
            logging.warning(f"Ocurrió un error al guardar el modelo: {e}")
            return 0

    @staticmethod
    def generate_layer(n: int):
        return [Neuron(np.random.uniform(-1, 1)) for _ in range(n)]

    @staticmethod
    def generate_neural_network(n_input: int, n_output: int, layers: list[int]):
        n = [Model.generate_layer(n_input)]
        for l in layers:
            n.append(Model.generate_layer(l))
        n.append(Model.generate_layer(n_output))

        for i in range(1, len(n)):
            for neuron in n[i]:
                neuron.inputs = [
                    Connection(np.random.uniform(-1, 1), prev)
                    for prev in n[i - 1]
                ]
        return n

    @staticmethod
    def new(input_neurons: int, output_neurons: int, hidden_layers: list[int], activation_function="ReLU"):
        model = Model()
        model.neural_network = Model.generate_neural_network(
            input_neurons, output_neurons, hidden_layers
        )
        model.activation_function = activation_function
        return model

    def activation_prime(self, z):
        if self.activation_function == "ReLU":
            return 1.0 if z > 0 else 0.0
        if self.activation_function == "LeakyReLU":
            return 1.0 if z > 0 else 0.01
        if self.activation_function == "Sigmoid":
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        return 1.0

    def activation(self, z):
        if self.activation_function == "ReLU":
            return z if z > 0 else 0
        if self.activation_function == "LeakyReLU":
            return z if z > 0 else 0.01 * z
        if self.activation_function == "Sigmoid":
            return 1 / (1 + np.exp(-z))
        return z

    def activate(self, inputs, bias):
        W = np.array([e.weight for e in inputs])
        A = np.array([e.neuron.a for e in inputs])
        z = np.sum(W * A) + bias
        a = self.activation(z)
        return z, a

    def train(self, dataset, epochs, learning_rate_w, learning_rate_b):
        for epoch in range(epochs):
            np.random.shuffle(dataset)

            for x, target in dataset:
                self.predict(x)

                A = [np.array([n.a for n in layer]) for layer in self.neural_network]
                Z = [np.array([n.z for n in layer]) if i > 0 else None
                     for i, layer in enumerate(self.neural_network)]

                L = len(self.neural_network) - 1
                d = [None] * (L + 1)

                y = np.array(target)
                y_hat = A[-1]
                d[L] = (y_hat - y) * np.array([self.activation_prime(z) for z in Z[L]])
                print(f"Error: {y - y_hat} | Epoch: {epoch}")
                for l in range(L - 1, 0, -1):
                    W_next = np.array([
                        [n_next.inputs[j].weight for j in range(len(self.neural_network[l]))]
                        for n_next in self.neural_network[l + 1]
                    ])

                    d[l] = (W_next.T @ d[l + 1]) * np.array(
                        [self.activation_prime(z) for z in Z[l]]
                    )

                for l in range(1, L + 1):
                    for i_n, neuron in enumerate(self.neural_network[l]):
                        neuron.bias -= learning_rate_b * d[l][i_n]

                        for j, connection in enumerate(neuron.inputs):
                            connection.weight -= learning_rate_w * d[l][i_n] * A[l - 1][j]

    def predict(self, data):
        if len(data) != len(self.neural_network[0]):
            logging.error("The entered data does not match the input values.")
            return

        for j, neuron in enumerate(self.neural_network[0]):
            neuron.a = data[j]

        for l in range(1, len(self.neural_network)):
            for neuron in self.neural_network[l]:
                z, a = self.activate(neuron.inputs, neuron.bias)
                neuron.z = z
                neuron.a = a

        return self.neural_network[-1]
