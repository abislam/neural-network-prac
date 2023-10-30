import math

#loss/cost function =  algorithm that quantifies how wrong a model is. Ideally want it to be 0.

#example softmax output
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0] #also known as ground truth

loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)