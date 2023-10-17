import numpy as np


#3 neurons 4 inputs

'''
Batches, Layers, and Objects


'''
#3 batches of 4 inputs
inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]



weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

#need to transpose weights matrix, otherwise shape error
#convert weights to numpy array
output = np.dot(inputs, np.array(weights).T) + biases
print(output)