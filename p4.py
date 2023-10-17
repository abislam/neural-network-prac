import numpy as np


np.random.seed(0)

'''
Batches, Layers, and Objects
'''
#3 batches of 4 inputs

#initial inputs
#3 neurons 4 inputs
#Sample training data. Variable name X is a convention for training data in Neural Networks.
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #generate weights based on range of #inputs and #neurons
        self.biases = np.zeros((1, n_neurons)) # 1 * n_neurons matrix of biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5) # we know from X that initial number of inputs is 4, number of neurons is up to you
layer2 = Layer_Dense(5,2) # since layer1 had 5 neurons, layer 2 will have 5 inputs, again, number of neurons is up to you

layer1.forward(X) #forward propogate layer1
print(layer1.output)
layer2.forward(layer1.output) #after layer1 forward propogation, forward propogate layer 2. In this case its the final output
print(layer2.output)