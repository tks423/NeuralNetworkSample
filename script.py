# this code defines a simple neural network with one layer, trains it using a basic dataset, and tests it with a new
# input. The network uses the tanh activation function and its derivative for training.

from numpy import array, random, tanh, dot

class NeuralNetworkSample():

    # Initializes the neural network with a random weight matrix of size 3x1.
    # The random seed is set to 1 for reproducibility.
    # learning_rate to control the size of weight adjustments. This can improve the stability and convergence
    # of the training process.
    def __init__(self, learning_rate=0.01):
        random.seed(1)
        self.weight_matrix = 2 * random.random((3, 1)) - 1
        self.learning_rate = learning_rate

    # Defines the hyperbolic tangent activation function.
    def tanh(self, x):
        return tanh(x)


    # Defines the derivative of the tanh function, which is used during backpropagation.
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2


    # Computes the output of the neural network by applying the tanh activation function to
    # the dot product of the inputs and the weight matrix.
    def forward_propagation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))


    # Trains the neural network using the provided training inputs and outputs. The training process involves forward
    # propagation, calculating the error, and adjusting the weights using the derivative of the tanh function.
    def train(self, train_inputs, train_outputs, num_train_iterations):
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)
            error = train_outputs - output
            adjustment = dot(train_inputs.T, error * self.tanh_derivative(output))
            self.weight_matrix += self.learning_rate * adjustment


# Creates an instance of the neural network, prints the initial random weights, trains the network with sample inputs
# and outputs, prints the new weights after training, and tests the network with a new input example.
if __name__ == "__main__":
    neural_network = NeuralNetworkSample()

    print('Random weights at the start of training')
    print(neural_network.weight_matrix)

    train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(train_inputs, train_outputs, 1000)

    print('New weights after training')
    print(neural_network.weight_matrix)

    # Test the neural network with a new situation.
    print("Testing network on new examples ->")
    print(neural_network.forward_propagation(array([1, 0, 0])))
