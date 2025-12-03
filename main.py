import numpy as np 
import logging
import uuid

DEBUG = True

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[%(levelname)s] %(message)s"
)

class Neuron:
    """
    Represents a neuron in a neural network.

    Attributes:
        inputs (list[Connection]): List of input connections to this neuron.
        bias (float): Bias value added to the neuron's weighted input sum.
        value (float): The current value/output of the neuron after activation.
    """
    def __init__(self, bias: float = 0):
        """
        Initializes a neuron with an optional bias.

        Args:
            bias (float): Initial bias value. Default is 0.
        """
        self.inputs: list[Connection] = []
        self.bias = bias
        self.value: float = 0
        logging.debug(f"Neuron created: bias={bias}")

    def __str__(self):
        """
        Returns a string representation of the neuron value inside parentheses.

        Returns:
            str: The value of the neuron as a string.
        """
        return f"({self.value})"
    
    def __repr__(self):
        """
        Returns: 
            str: A string representation suitable for debugging/lists, showing the value.
        """
        return f"({self.value})"

class Connection:
    """
    Represents a directed, weighted connection from one neuron to another.

    Attributes:
        weight (float): The multiplicative weight for the connection.
        neuron (Neuron): The source neuron for this connection.
    """
    def __init__(self, weight: float, neuron: Neuron):
        """
        Initializes a connection to a neuron with a given weight.

        Args:
            weight (float): Weight for this connection.
            neuron (Neuron): The source neuron whose value is used.
        """
        self.weight = weight
        self.neuron = neuron        
        
        logging.debug(f"Connection created: weight={weight}, neuron={neuron.value}")

class Model:
    """
    Encapsulates a simple feedforward neural network with one or more layers.

    Attributes:
        neural_network (list[list[Neuron]]): Nested list of layers, each containing neurons.
        activation_function (str): Name of the activation function used ("ReLU", "Sigmoid", etc.).
    """
    def __init__(self):
        """
        Initializes the neural network model.

        Attributes:
            neural_network (list[list[Neuron]]): The architecture/layers of the network.
            activation_function (str): Desired activation function for neurons.
        """
        self.neural_network: list[list[Neuron]] = []
        self.activation_function = ""
        
        logging.debug("Model initialized.")
        
    def save(self):
        logging.debug("Estructurando modelo para ser guardado...")
        model_data = [[ [n.bias, [e.weight for e in n.inputs]] for n in layer ] for layer in self.neural_network]
        
        logging.debug(f"Datos del modelo estructurados: {model_data}")
        name = f"{uuid.uuid1()}.73nn"
        try:
            logging.debug("Intenando guardar el modelo...")
            with open(name, 'w', encoding='utf-8') as model:
                model.write(f"{model_data}")
                logging.info(f"Modelo guardado exitosamente como {name}")
                return 1
        except Exception as e:
            logging.warning(f"Ocurrió un error al guardar el modelo: {e}")
            return 0
        
    @staticmethod
    def generate_layer(n: int):
        """
        Generates a layer containing n neurons with random biases.

        Args:
            n (int): Number of neurons in the layer.

        Returns:
            list[Neuron]: List of Neuron objects.
        """
        layer = [Neuron(np.random.uniform(-1,1)) for _ in range(n)]
        logging.debug(f"Layer generated ({n} neurons).")
        return layer

    @staticmethod
    def generate_neural_network(n_entrada: int, layers: list[int]):
        """
        Creates a neural network architecture with a specified number of input neurons, hidden layers, and an output neuron.

        Args:
            n_entrada (int): Number of input neurons.
            layers (list[int]): A list specifying the number of neurons in each hidden layer.

        Returns:
            list[list[Neuron]]: The created neural network as a list of layers.
        """
        n = [Model.generate_layer(n_entrada)]
        for l in layers:
            n.append(Model.generate_layer(l))
        n.append(Model.generate_layer(1))

        for i, layer in enumerate(n):
            if i != 0:
                # Connect each neuron to every neuron in the previous layer
                for neuron in layer:
                    neuron.inputs = [Connection(np.random.uniform(-1,1), a_neuron) for a_neuron in n[i-1]]
        
        logging.debug(f"Neural network generated: structure {[len(layer) for layer in n]}")
        return n
    
    @staticmethod
    def new(input_neurons: int, hidden_layers: list[int], activation_function:str = "LeakyReLU"):
        """
        Creates a new Model instance with the specified architecture and activation function.

        Args:
            input_neurons (int): Number of input neurons.
            hidden_layers (list[int]): Number of neurons in each hidden layer.
            activation_function (str): Name of the activation function to use.

        Returns:
            Model: The constructed Model instance.
        """
        model = Model()
        logging.debug(f"New Model: input_neurons={input_neurons}, hidden_layers={hidden_layers}, activation={activation_function}")
        model.neural_network = Model.generate_neural_network(input_neurons, hidden_layers)
        
        model.activation_function = activation_function
        
        return model

    def compress(self, n: float):
        """
        Applies the selected activation function to input 'n'.

        Args:
            n (float): Input value to the activation function.

        Returns:
            float: Output after activation function is applied.
        """
        if self.activation_function == "ReLU":
            result = 0 if n <= 0 else n
        if self.activation_function == "LeakyReLU":
            result = n if n >= 0 else 0.01 * n
        elif self.activation_function == "Sigmoid":
            result = 1 / (1 + np.exp(-n))
        elif self.activation_function == "Binary":
            result = 0 if n <= 0 else 1
        elif self.activation_function == "Sign":
            result = -1 if n <= 0 else 1
        else:
            logging.warning("Unknown compression function.")
            result = 0
        logging.debug(f"Compress: input={n}, output={result}, function={self.activation_function}")
        return result

    def activate(self, inputs: list[Connection], bias: float):
        """
        Calculates the activated value of a neuron given its inputs and bias.

        Args:
            inputs (list[Connection]): Connections providing input values and weights.
            bias (float): Bias to be added after summing weighted inputs.

        Returns:
            float: Activated value (after applying activation function).
        """
        W = np.array([e.weight for e in inputs])                # Array of weights
        A = np.array([e.neuron.value for e in inputs])          # Array of input neuron values
        z = np.sum(W * A) + bias                               # Weighted sum + bias
        act = self.compress(z)
        logging.debug(f"Activating Neuron: inputs={A.tolist()}, weights={W.tolist()}, bias={bias}, z={z}, output={act}")
        return act

    def train(self, dataset: list[tuple[list[float], float]], epochs=500, learning_rate_w=0.001, learning_rate_b=0.01):
        """
        Trains the network using a simple gradient-like update (not backprop).

        Args:
            dataset (list[tuple[list[float], float]]): List of (input, expected_output) pairs.
            epochs (int): Number of times to iterate over the dataset.
            learning_rate_w (float): Learning rate for weights.
            learning_rate_b (float): Learning rate for biases.
        """
        logging.info(f"Starting training for {epochs} epochs.")
        for ep in range(epochs):
            for inp, expected_output in dataset:
                predicted_ouput = self.predict(inp)                # Perform forward pass
                e = expected_output - predicted_ouput.value        # Compute error
                logging.debug(f"Epoch={ep}, Input={inp}, Expected={expected_output}, Predicted={predicted_ouput.value}, Error={e}")
                
                # Update weights and biases for hidden/output layers
                for layer in self.neural_network[1:]:
                    for neuron in layer:
                        for enlace in neuron.inputs:
                            old_weight = enlace.weight
                            enlace.weight += learning_rate_w * e * enlace.neuron.value
                            logging.debug(f"Updating weight: {old_weight} -> {enlace.weight}")
                        old_bias= neuron.bias
                        neuron.bias += learning_rate_b * e
                        logging.debug(f"Updating bias: {old_bias} -> {neuron.bias}")

    def predict(self, data: list[float]):
        """
        Performs a forward pass through the network to make a prediction.

        Args:
            data (list[float]): Input values, one for each neuron in the input layer.

        Returns:
            Neuron or str: The output neuron (with value attribute) or error string if input size mismatched.
        """
        if len(data) != len(self.neural_network[0]): 
            logging.error("The entered data does not match the input values of the neural network.")
            return "The entered data does not match the input values of the neural network."

        for i, layer in enumerate(self.neural_network):
            if i == 0:
                # Set values for the input layer neurons
                for j, neuron in enumerate(layer):
                    neuron.value = data[j]
                    logging.debug(f"Asigned input layer: {data}")
            else:
                # Calculate activation for the rest layers
                for neuron in layer:
                    neuron.value = self.activate(neuron.inputs, neuron.bias)
                logging.debug(f"Layer {i} processed. Values={[neuron.value for neuron in layer]}")
        logging.info(f"Predicción final: {self.neural_network[-1][0]}")
        return self.neural_network[-1][0]    # Return final output neuron


# # Example usage: Build and train a simple network to convert Celsius to Fahrenheit
# n1 = Neuron()                                       # Input neuron
# n2 = Neuron(np.random.uniform(-1,1))                # Output neuron with random bias

# n2.inputs = [
#     Connection(np.random.uniform(-1,1), n1)         # Connect n1 to n2 with random weight
# ]

# model = Model([
#     [n1],      # Input layer
#     [n2],      # Output layer
# ], "LeakyReLU")

# # Prepare training dataset: pairs of Celsius and Fahrenheit values
# model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)

# print("Modelo entrenado.")
# print(model.predict([10]))
# print(model.predict([35]))

# model = Model.new(1, [])

# model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)
# print("Modelo entrenado.")
# print(model.predict([10]))
# print(model.predict([35]))