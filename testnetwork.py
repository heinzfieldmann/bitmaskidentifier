import numpy as np

# My trying to understand the magic of neural networks. Hans

# Define some shapes champ! Paint them in a 4x4 grid and make it into list.

# Empty (no shape)
#0000
#0000
#0000
#0000
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# I-shape
#1111
#0000
#0000
#0000
# [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]

# O-shape
#1100
#1100
#0000
#0000
# [1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]

# T-shape
#1110
#0100
#0000
#0000
# List
# [1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0]

# J-shape
#0110
#0100
#1100
#0000
# [0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,0]

# L-shape (reflection of J)
#1000
#1000
#1100
#0000
# [1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0]

# S-shape
#0110
#1100
#0000
#0000
# [0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0]

# Z-shape
#1100
#0110
#0000
#0000
# [1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0]

# Define the Tetris shapes in a list.
# 7 one-sided tetromineos cause they show up in Tetris
# 
tetris_shapes = [
    [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],  # I shape
    [1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],  # O shape
    [1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0],  # T shape
    [1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0],  # L shape
    [0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,0],  # J shape
    [0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],  # S shape
    [1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0]   # Z shape
]

# Later: try to see if the network can recognize when the shapes are rotated/translated.

# Function to print the shapes as 4x4 grids
def print_shape(shape):
    for i in range(4):  # 4 rows
        row = shape[i*4:(i+1)*4]  # Select 4 elements for each row
        print(' '.join(str(x) for x in row))
    print(" ")  # Add a new line between shapes

# Iterate over each Tetris shape and print it
for shape in tetris_shapes:
    print_shape(shape)



# Classify the outputs
# Labels for each shape (one-HOT encoding)
tetris_labels = [
    [1,0,0,0,0,0,0],  # I shape
    [0,1,0,0,0,0,0],  # O shape
    [0,0,1,0,0,0,0],  # T shape
    [0,0,0,1,0,0,0],  # L shape
    [0,0,0,0,1,0,0],  # J shape
    [0,0,0,0,0,1,0],  # S shape
    [0,0,0,0,0,0,1]   # Z shape
]

# Neural Network parameters
input_size = 16 # one for each "pixel"
hidden_size = 8
output_size = 7 # One for each label.
learning_rate = 0.01

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
print(type(W1))
print(W1)
exit()
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Softmax function for output
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward propagation
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = softmax(Z2)
    return output

# Backpropagation and training will follow...
