import math

#softmax activation


layer_outputs = [4.8, 1.21, 2.385]


#applying exponentiation. handles negative data without losing meaning of said data

E = math.e #eulers nhumber

exp_values = []

for output in layer_outputs:
        exp_values.append (E ** output) # ** means to the power
print('exponentiated values:')
print(exp_values)