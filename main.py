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

        hidden_layers_neurons = (
            [len(layer) for layer in model_data[1:]] if len(model_data) >= 3 else []
        )

        model = Model.new(first_layer_neurons, hidden_layers_neurons)

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

        model_data_string = f"{json.dumps(model_data)}".encode("utf-8")

        try:
            with open(name, "wb") as model:
                model.write(zlib.compress(model_data_string, 9))
                logging.info(f"Modelo guardado exitosamente como {name}")
                return 1
        except Exception as e:
            logging.warning(f"Ocurrió un error al guardar el modelo: {e}")
            return 0

    @staticmethod
    def generate_layer(n: int):
        layer = [Neuron(np.random.uniform(-1, 1)) for _ in range(n)]
        return layer

    @staticmethod
    def generate_neural_network(n_entrada: int, layers: list[int]):
        n = [Model.generate_layer(n_entrada)]
        if not len(layers) == 0:
            for l in layers:
                n.append(Model.generate_layer(l))
        n.append(Model.generate_layer(1))

        for i, layer in enumerate(n):
            if i != 0:
                # Connect each neuron to every neuron in the previous layer
                for neuron in layer:
                    neuron.inputs = [
                        Connection(np.random.uniform(-1, 1), a_neuron)
                        for a_neuron in n[i - 1]
                    ]

        return n

    @staticmethod
    def new(
        input_neurons: int,
        hidden_layers: list[int],
        activation_function: str = "ReLU",
    ):
        model = Model()
        model.neural_network = Model.generate_neural_network(
            input_neurons, hidden_layers
        )

        model.activation_function = activation_function

        return model

    def activation_prime(self, z: float):
        if self.activation_function == "ReLU":
            return 1.0 if z > 0 else 0.0
        if self.activation_function == "LeakyReLU":
            return 1.0 if z > 0 else 0.01
        if self.activation_function == "Sigmoid":
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)

        return 1.0

    def activation(self, z: float):
        if self.activation_function == "ReLU":
            return z if z > 0 else 0
        if self.activation_function == "LeakyReLU":
            return z if z > 0 else 0.01 * z
        if self.activation_function == "Sigmoid":
            return 1 / (1 + np.exp(-z))
        return z

    def activate(self, inputs: list[Connection], bias: float):
        W = np.array([e.weight for e in inputs])  # Array of weights
        A = np.array([e.neuron.a for e in inputs])  # Array of input neuron values
        z = np.sum(W * A) + bias  # Weighted sum + bias
        a = self.activation(z)
        return (z, a)

    def train(self, dataset: list, epochs: int, learning_rate_w: float, learning_rate_b: float):
        pass

    def predict(self, data: list[float]):
        if len(data) != len(self.neural_network[0]):
            logging.error(
                "The entered data does not match the input values of the neural network."
            )
            return "The entered data does not match the input values of the neural network."

        for i_l, layer in enumerate(self.neural_network):
            if i_l == 0:
                # Set values for the input layer neurons
                for j, neuron in enumerate(layer):
                    neuron.a = data[j]
            else:
                # Calculate activation for the rest layers
                for neuron in layer:
                    Zn, An = self.activate(neuron.inputs, neuron.bias)
                    neuron.a = An
                    neuron.z = Zn
        logging.info(f"Predicción final: {self.neural_network[-1][0].a}")
        return self.neural_network[-1][0].a  # Return final output neuron
