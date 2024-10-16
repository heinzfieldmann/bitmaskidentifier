#!/usr/local/bin/python3

import numpy as np

# My trying to understand the magic of neural networks. Hans

# Prova nu att rotera eller gör om formen till en matris istället för 16 1-dimensionell. Enklare att rotera en matris
# Skapa en testkörning som visar hur mycket rätt nätverket har.

# I could create a random shape and see what the highest probability is for that shape since it has to "choose" one right?


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
#0100
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

# Z-shape (reflection of s)
#1100
#0110
#0000
#0000
# [1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0]

# Define the Tetris shapes in a list.
# 7 one-sided tetromineos cause they show up in Tetris
# 
tetris_shapes = [
    [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],  # I Shape
    [1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],  # O shape
    [1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0],  # T shape
    [1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0],  # L shape
    [0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0],  # J shape
    [0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],  # S shape
    [1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0]   # Z shape
]

# Later: try to see if the network can recognize when the shapes are rotated/translated.
# This function will rotate the 
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
output_size = 7 # One for each label. Any shape.
# Hyperparameter that will say that the gradient is adjusted by 1%
learning_rate = 0.01

# W1: Initialize weights and biases
# Initialise the matrice with random weigthts between the 16 input and the hidden layer (8 at the moment)
# W1 will contains a matrix of 16*8 (128 elements
# Rows 16 (for each input) and Columns 8 (for the hidden layer)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
# Bias and weight for the output thingy.
# So the matrix has to be 8*7 because we have 7 outputs to be able to label the shapes.
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation function. relu
# np.maximum can eat lists. max() we have to iterate over?
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

# Categorical Cross-Entropy Loss
def cross_entropy_loss(predictions, labels):
    # Small epsilon value to avoid log(0)
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(labels * np.log(predictions)) / predictions.shape[0]

# Backpropagation
def back_propagation(X, A1, output, Y):
    global W1, b1, W2, b2
    
    m = X.shape[0]  # Number of samples
    
    # Calculate the error at the output layer
    dZ2 = output - Y  # Difference between prediction and actual label
    dW2 = np.dot(A1.T, dZ2) / m  # Gradient of W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Gradient of b2
    
    # Calculate the error at the hidden layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (A1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X.T, dZ1) / m  # Gradient of W1
    dB1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Gradient of B1
    
    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * dB1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Training loop
def train(X, Y, epochs=1000):
    for i in range(epochs):
        # Forward propagation
        output = forward_propagation(X)
        
        # Compute loss
        loss = cross_entropy_loss(output, Y)
        
        # Backward propagation
        Z1 = np.dot(X, W1) + b1  # Hidden layer linear part
        A1 = relu(Z1)  # Hidden layer activation
        back_propagation(X, A1, output, Y)
        
        # Print loss every 100 epochs
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

# Convert the tetris shapes and labels into NumPy arrays
X_train = np.array(tetris_shapes)  # Input data (Tetris shapes)
Y_train = np.array(tetris_labels)  # One-hot encoded labels

# Train the network
train(X_train, Y_train, epochs=10000)    

def predict_shape(shape):
    # Convert the shape into a numpy array and reshape to match input dimensions
    shape = np.array(shape).reshape(1, -1)  # Reshape to 1x16 array
    
    # Perform forward propagation to get the prediction
    output = forward_propagation(shape)
    
    # Get the index of the highest probability in the output
    predicted_class = np.argmax(output)
    
    # Map the index to the corresponding Tetris shape label
    shape_labels = ["I", "O", "T", "L", "J", "S", "Z"]
    
    return shape_labels[predicted_class], output

# shape_labels should be global or in a data set.
# Hm. sätt ihop labels så vi vet vilken index vi använder? Samt skriva ut prob för varje label.
# Convert this could be a function mate..... 
for shape in tetris_shapes:
    print()
    print("Next try: this shape below:")
    print_shape(shape)
    print("Can you predict this shape Mr. computer?")
    predicted_shape, probabilities = predict_shape(shape)
    shape_labels = ["I", "O", "T", "L", "J", "S", "Z"]
    print(f"Predicted Shape: {predicted_shape}")
    print(f"Probabilities: {probabilities}")
    for label in shape_labels:
        print(label)
        print(probabilities[0])

print("random shape")
random_shape = np.random.choice([0, 1], size=16)
print_shape(random_shape)
predicted_shape, probabilities = predict_shape(random_shape)
print(f"Predicted Shape: {predicted_shape}")
print(f"Probabilities: {probabilities}")
