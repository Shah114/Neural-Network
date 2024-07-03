'''Importing Modules'''
from joblib.numpy_pickle_utils import xrange
from numpy import *

'''Making Class'''
class NeuralNet(object):
    def __init__(self):
        # Generate random numbers
        random.seed(1)

        # Assign random weights to a 3 x 1 matrix
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network and adjust the weights each time.
    def train(self, inputs, outputs, trainin_iterations):
        for iteration in xrange(trainin_iterations):
            # Pass the training set through the network.
            output = self.learn(inputs)

            # Calculate the error
            error = outputs - output

            # Adjust the weights by a factor
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += factor

        # The neural network thinks.
    
    def learn(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
    
'''Main part'''
if __name__ == "__main__":
    # Initialize
    neural_network = NeuralNet()

    # The Training set
    inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    outputs = array([[1, 0, 1]]).T  

    # Train the neural network
    neural_network.train(inputs, outputs, 10000)

    # Test the neural network with a test example.
    print("Result is:")
    print(neural_network.learn(array([1, 0, 1])))