import numpy as np
from SetCoverPy import setcover

# seed numpy
np.random.seed(0)

from itertools import combinations


def gen_set_cover(num_matrices, max_universe_size):
    X = []
    y = []
    for _ in range(num_matrices):
        # Generate a random binary matrix
        num_rows = np.random.randint(3, max_universe_size + 1)
        num_cols = np.random.randint(1, max_universe_size + 1)
        binary_matrix = np.zeros((num_rows, num_cols), dtype=bool)
        for i in range(num_rows):
            binary_matrix[i, np.random.randint(0, num_cols)] = True
        for j in range(num_cols):
            binary_matrix[np.random.randint(0, num_rows), j] = True

        # Find a set cover solution using brute force
        solution_indices = set_cover_bruteforce(binary_matrix)
        # Convert solution indices to the actual subsets
        solution = [binary_matrix[:, idx] for idx in solution_indices]

        # Convert binary matrix and solution to sequences
        input_sequence = [list(np.where(column)[0]) for column in binary_matrix.T]
        # Map the solution subsets to their corresponding indices in the input sequence
        label_sequence = [subset for idx, subset in enumerate(input_sequence) if idx in solution_indices]

        X.append(input_sequence)
        y.append(label_sequence)

    return X, y


def is_covering(matrix, selected_columns, rows):
    covered = set()
    for col in selected_columns:
        covered.update(np.where(matrix[:, col])[0])
    return len(covered) == rows


def set_cover_bruteforce(matrix):
    rows, cols = matrix.shape
    for num_columns in range(1, cols + 1):
        for selected_columns in combinations(range(cols), num_columns):
            if is_covering(matrix, selected_columns, rows):
                return selected_columns
    return None


# Example usage:
num_matrices = 10
max_universe_size = 6
X, y = gen_set_cover(num_matrices, max_universe_size)

# Print the generated sequences
print(X)
print(y)


