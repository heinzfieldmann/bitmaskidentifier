import numpy as np


v = [1.0,-10.0,0.0,15.0,-2.0]


def sig(x):
    return 1/(1+np.exp(-x))


def relu(x):
#the relu truncates all values less than 0
    return max(0,x)

for x in v:
    print('Applying SIGMOID Activation on (%.1f) gives %.1f' % (x, sig(x)))
    print('Applying RELU Activation on (%.1f) gives %.1f' % (x, relu(x)))




