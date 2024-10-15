import numpy as np

# Example shape (flattened 1D list)
shape_1d = [0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0]  # I-shape

# Convert 1D list to 2D matrix (4x4)
def convert_to_2d(matrix_1d):
    return [matrix_1d[i:i+4] for i in range(0, len(matrix_1d), 4)]

# Function to rotate the 2D matrix 90 degrees clockwise
def rotate_clockwise(matrix_2d):
    return [list(reversed(col)) for col in zip(*matrix_2d)]

# Convert the 2D matrix back to 1D list
def flatten_to_1d(matrix_2d):
    return [item for row in matrix_2d for item in row]

# Main function to rotate the shape
def rotate_1d_shape(shape_1d):
    # Step 1: Convert to 2D
    matrix_2d = convert_to_2d(shape_1d)
    
    # Step 2: Rotate the 2D matrix
    rotated_matrix = rotate_clockwise(matrix_2d)
    
    # Step 3: Convert back to 1D
    return flatten_to_1d(rotated_matrix)

# Perform the rotation
rotated_shape_1d = rotate_1d_shape(shape_1d)

# Print the result
print(rotated_shape_1d)

