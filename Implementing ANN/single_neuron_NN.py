'''A single neuron neural network in Python'''

# Import all necessary libraries
from numpy import exp, array, random, dot, tanh

'''Making Class'''
class NeuralNetwork():
    '''Class to create a neural network with single neuron'''
    def __init__(self):
        # Using seed to make sure it'll
        # generate same weights in every run
        random.seed(1)

        # 3x1 Weight matrix
        self.weight_matrix = 2 * random.random((3, 1)) - 1

    # tanh as activation function
    def tanh(self, x):
        return tanh(x)
    
    # Derivative of tanh function.
    # Needed to calculate the gradients.
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2
    
    # Forward propagation
    def forward_propogation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))

    # Training the neural network.
    def train(self, train_inputs, train_outputs,
                            num_train_iterations):
        
        # Number of iterations we want to
        # perform for this set of input.
        for iteration in range(num_train_iterations):
            output = self.forward_propogation(train_inputs)

            # Calculate the error in the output.
            error = train_outputs - output

            # multiply the error by input and then
            # by gradient of tanh function to calculate
            # the adjustment needs to be made in weights
            adjustment = dot(train_inputs.T, error *
                             self.tanh_derivative(output))
            
            # Adjust the weight matrix
            self.weight_matrix += adjustment

'''Driver Code'''
if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print('Random weights at the start of training')
    print(neural_network.weight_matrix)

    train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(train_inputs, train_outputs, 10000)

    print ('New weights after training')
    print(neural_network.weight_matrix)

    # Test the neural network with a new situation.
    print ("Testing network on new examples ->")
    print(neural_network.forward_propogation(array([1, 0, 0])))