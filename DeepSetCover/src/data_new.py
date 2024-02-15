import numpy as np
from SetCoverPy import setcover

# seed numpy
np.random.seed(0)


def generate_binary_matrices2(num_matrices, max_universe_size):
    binary_matrices = []

    for _ in range(num_matrices):
        # Generate a random number of rows and columns for each matrix
        num_rows = np.random.randint(3, max_universe_size + 1)  # min universe size is 3
        num_cols = np.random.randint(1, max_universe_size + 1)

        # Create a random binary matrix with at least one True value in each row
        binary_matrix = np.zeros((num_rows, num_cols), dtype=bool)
        for i in range(num_rows):
            # Ensure at least one True value in each row
            binary_matrix[i, np.random.randint(0, num_cols)] = True

            # Ensure at least one True value in each column
        for j in range(num_cols):
            binary_matrix[np.random.randint(0, num_rows), j] = True

        binary_matrices.append(binary_matrix)

    return binary_matrices


# define parameters
num_matrices = 10
max_universe_size = 6

# Generate binary matrices
generated_matrices = generate_binary_matrices2(num_matrices, max_universe_size)

# Print the generated matrices
for i, matrix in enumerate(generated_matrices):
    print(f"Generated Binary Matrix {i + 1}:")
    print(matrix)
    print()


# Function to convert a binary matrix to SetCoverPy input format
def convert_to_setcover_input(binary_matrix):
    return binary_matrix, np.ones(binary_matrix.shape[1], dtype=float)


# List to store SetCoverPy solutions
setcover_solutions = []

# Iterate over the matrices and find set cover solutions
for i, generated_matrix in enumerate(generated_matrices):
    # Convert binary matrix to SetCoverPy input format
    setcover_input = convert_to_setcover_input(generated_matrix)

    # Instantiate SetCover object
    g = setcover.SetCover(*setcover_input)

    # Run the solver
    solution, time_used = g.SolveSCP()

    # Save the solution to the list
    setcover_solutions.append(g.s)

    # Print the solution and total cost
    print(f"Set Cover Solution for Matrix {i + 1}:")
    print("Solution (g.s):", g.s)
    print("Total Cost of the Set Cover:", g.total_cost)
    print()

# Now 'setcover_solutions' contains the binary vectors of solutions for each matrix

# Print the setcover_solutions list separately
for i, solution in enumerate(setcover_solutions):
    print(f"Set Cover Solution for Matrix {i + 1}:")
    print("Solution (g.s):", solution)
    print()


def convert_to_sequences(binary_matrices, solutions):
    sequences = []

    for binary_matrix, solution in zip(binary_matrices, solutions):
        # Convert binary matrix to a sequence of subsets
        input_sequence = [list(np.where(column)[0]) for column in binary_matrix.T]

        # Convert solution to a sequence of integers
        # label_sequence = list(np.where(solution)[0])
        # Convert solution to a sequence of integers
        label_sequence = [index for index in np.where(solution)[0]]

        sequences.append((input_sequence, label_sequence))

    return sequences


# (Assuming 'generated_matrices' contains the binary matrices and 'setcover_solutions' contains the SetCoverPy solutions)
# Input is the list of matrices, output is the list of integers

# Convert binary matrices and solutions to sequences
sequences = convert_to_sequences(generated_matrices, setcover_solutions)

# Print the converted sequences
for i, (input_sequence, label_sequence) in enumerate(sequences):
    print(f"Sequence {i + 1} - Input:")
    print(input_sequence)
    print("Label:")
    print(label_sequence)
    print()
