import math
import numpy as np

#softmax activation

#dummy values
layer_outputs = [4.8, 1.21, 2.385]


#applying exponentiation. handles negative data without losing meaning of said data
#important for calculating probabilities.

#exponentiation using numpy
# for loop, for each value in vector layer_outputs, exponentiate the value, store in exp_values
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

#normalization of exp_values using numpy
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))