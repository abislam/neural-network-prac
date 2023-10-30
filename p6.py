
import numpy as np

#softmax activation

#dummy values
layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]


#applying exponentiation. handles negative data without losing meaning of said data
#important for calculating probabilities.

#exponentiation using numpy
# for loop, for each value in vector layer_outputs, exponentiate the value, store in exp_values
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

#normalization of exp_values using numpy

#axis 0 takes sum of columns, not what we want
#print(np.sum(layer_outputs, axis = 0))
 
#axis 1 takes sum of rows, this is what we want, keepdims true ensures same dimensions as input
#print(np.sum(layer_outputs, axis = 1, keepdims = True)) 

norm_values = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
print(norm_values)