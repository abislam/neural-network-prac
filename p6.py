import math

#softmax activation


layer_outputs = [4.8, 1.21, 2.385]


#applying exponentiation. handles negative data without losing meaning of said data
#important for calculating probabilities.

E = math.e #eulers nhumber

exp_values = []

for output in layer_outputs:
        exp_values.append (E ** output) # ** means to the power
print('exponentiated values:')
print(exp_values)

#normalization of exp_values
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
        norm_values.append(value / norm_base)
print('Normalized exponentiated values:')
print(norm_values)

print('Sum of normalized values:', sum(norm_values))
