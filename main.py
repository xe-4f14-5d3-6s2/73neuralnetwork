import numpy as np 

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

    def __str__(self):
        """Displays neuron value between bars."""
        return f"| {self.value} |"
    
    def __repr__(self):
        """Same as __str__ for representation in lists/debugging."""
        return f"| {self.value} |"

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

class Model:
    """
    Encapsulates a simple feedforward neural network with one or more layers.

    Attributes:
        neural_network (list[list[Neuron]]): Nested list of layers, each containing neurons.
        activation_function (str): Name of the activation function used ("ReLU", "Sigmoid", etc.).
    """
    def __init__(self, neural_network: list[list[Neuron]], activation_function:str = "LeakyReLU"):
        """
        Initializes the neural network model.

        Args:
            neural_network (list[list[Neuron]]): The architecture/layers of the network.
            activation_function (str): Desired activation function for neurons.
        """
        self.neural_network = neural_network
        self.activation_function = activation_function

    def compress(self, n: float):
        """
        Applies the selected activation function to input 'n'.

        Args:
            n (float): Input value to the activation function.

        Returns:
            float: Output after activation function is applied.
        """
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
        return self.compress(z)

    def train(self, dataset: list[tuple[list[float], float]], epochs=500, learning_rate_w=0.001, learning_rate_b=0.01):
        """
        Trains the network using a simple gradient-like update (not backprop).

        Args:
            dataset (list[tuple[list[float], float]]): List of (input, expected_output) pairs.
            epochs (int): Number of times to iterate over the dataset.
            learning_rate_w (float): Learning rate for weights.
            learning_rate_b (float): Learning rate for biases.
        """
        for _ in range(epochs):
            for inp, expected_output in dataset:
                predicted_ouput = self.predict(inp)                # Perform forward pass
                e = expected_output - predicted_ouput.value        # Compute error
                
                # Update weights and biases for hidden/output layers
                for layer in self.neural_network[1:]:
                    for neuron in layer:
                        for enlace in neuron.inputs:
                            enlace.weight += learning_rate_w * e * enlace.neuron.value
                        neuron.bias += learning_rate_b * e

    def predict(self, data: list[float]):
        """
        Performs a forward pass through the network to make a prediction.

        Args:
            data (list[float]): Input values, one for each neuron in the input layer.

        Returns:
            Neuron or str: The output neuron (with value attribute) or error string if input size mismatched.
        """
        if len(data) != len(self.neural_network[0]): 
            return "Los datos no coinciden con los parámetros de inp."

        for i, layer in enumerate(self.neural_network):
            if i == 0:
                # Set values for the input layer neurons
                for i, neuron in enumerate(layer):
                    neuron.value = data[i]
            else:
                # Calculate activation for the rest layers
                for i, neuron in enumerate(layer):
                    neuron.value = self.activate(neuron.inputs, neuron.bias)

        return self.neural_network[-1][0]    # Return final output neuron


# Example usage: Build and train a simple network to convert Celsius to Fahrenheit
n1 = Neuron()                                       # Input neuron
n2 = Neuron(np.random.uniform(-1,1))                # Output neuron with random bias

n2.inputs = [
    Connection(np.random.uniform(-1,1), n1)         # Connect n1 to n2 with random weight
]

model = Model([
    [n1],      # Input layer
    [n2],      # Output layer
], "LeakyReLU")

# Prepare training dataset: pairs of Celsius and Fahrenheit values
model.train([([c], ((c*(9/5))+32)) for c in range(500)], 1000, 0.000001, 0.001)

print("Modelo entrenado.")
print(model.predict([10]))
print(model.predict([35]))