import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

np.random.seed(0)


#Sample training data. Variable name X is a convention for training data in Neural Networks.
#Lower case y is the target or classifications
X, y = spiral_data(100, 3) #100 feature sets of 3 classes

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #generate weights based on range of #inputs and #neurons
        self.biases = np.zeros((1, n_neurons)) # 1 * n_neurons matrix of biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


'''
#ReLU Logic Example
    for i in inputs:
            if i > 0:
                output.append(i)
            elif i <= 0:
                output.append(0)
'''
class Activation_ReLU:
     def forward(self, inputs):
          self.output = np.maximum(0, inputs)

'''
Softmax Activation
'''
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

layer1 = Layer_Dense(2,5) # number of inputs, number of neurons
activation1 = Activation_ReLU()

layer1.forward(X) #forward propogate layer1
activation1.forward(layer1.output) #apply ReLU to forward prop outputs of layer1
print(activation1.output)