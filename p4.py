import numpy as np


'''
Batches, Layers, and Objects
'''
#3 batches of 4 inputs

#initial inputs
#3 neurons 4 inputs
inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


#1st layer
weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2.0, 3.0, 0.5]

#2nd layer - only 3 weights because the outputs of the first layer went to 3 neurons in the second layer
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

#need to transpose weights matrix, otherwise shape error
#convert weights to numpy array
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2 #output of layer1 became input of layer2

print(layer2_outputs)