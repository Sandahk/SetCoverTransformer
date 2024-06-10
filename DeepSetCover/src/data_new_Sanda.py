import numpy as np
from itertools import combinations

# seed numpy
np.random.seed(100)

def generate_setcover_sequences(num_matrices, max_universe_size):
    sequences = []
    universe_list = list(range(13))  # Default universe list from 0 to 20
    selected_elements = np.random.choice(universe_list, max_universe_size, replace=False)
    print(selected_elements)


    for _ in range(num_matrices):
        # Generate a random binary matrix
        num_rows = np.random.randint(2, max_universe_size) #number of subsets
        num_cols = np.random.randint(3, max_universe_size + 1) #elements
        binary_matrix = np.zeros((num_rows, num_cols), dtype=bool)

        for i in range(num_rows):
            # Set a random True value in the row
            binary_matrix[i, np.random.randint(num_cols)] = True
        for j in range(num_cols):
            # Set a random True value in the column
            binary_matrix[np.random.randint(num_rows), j] = True


        # Find a set cover solution using brute force
        solution_indices = set_cover_bruteforce(binary_matrix)
        # Convert solution indices to the actual subsets
        solution = [binary_matrix[:, idx] for idx in solution_indices]

        # Convert binary matrix and solution to sequences
        #input_sequence = [list(np.where(column)[0]) for column in binary_matrix.T]
        # Convert binary matrix columns to sequences of selected elements
        input_sequence = []
        for column in binary_matrix.T:
            subset = [selected_elements[row] for row in np.where(column)[0]]
            input_sequence.append(subset)


        # Map the solution subsets to their corresponding indices in the input sequence
        label_sequence = [subset for idx, subset in enumerate(input_sequence) if idx in solution_indices]
        sequences.append((input_sequence, label_sequence))

    return sequences


def generate_setcover_sequences2(num_matrices, max_universe_size):
    sequences = []
    universe_list = list(range(10))  # Default universe list from 0 to 20
    selected_elements = np.random.choice(universe_list, max_universe_size, replace=False)
    print(selected_elements)


    for _ in range(num_matrices):
        # Generate a random binary matrix
        num_rows = np.random.randint(3, max_universe_size + 1) #elements
        num_cols = np.random.randint(2, max_universe_size + 1) #sets
        binary_matrix = np.random.choice([False, True], size=(num_rows, num_cols))  # Randomly generate binary matrix

        # Ensure each row has at least one True value
        for i in range(num_rows):
            if not binary_matrix[i].any():
                binary_matrix[i, np.random.randint(0, num_cols)] = True

        # Ensure each column has a specific number of True values
        for j in range(num_cols):
            if not binary_matrix[:, j].any():
                binary_matrix[np.random.randint(0, num_rows), j] = True

        # Remove duplicate rows
        binary_matrix = np.unique(binary_matrix, axis=1)

        print("Binary matrix (no duplicates):")
        print(binary_matrix)

        # Find a set cover solution using brute force
        solution_indices = set_cover_bruteforce(binary_matrix)
        # Convert solution indices to the actual subsets
        solution = [binary_matrix[:, idx] for idx in solution_indices]

        # Convert binary matrix and solution to sequences
        #input_sequence = [list(np.where(column)[0]) for column in binary_matrix.T]
        # Convert binary matrix columns to sequences of selected elements
        input_sequence = []
        for column in binary_matrix.T:
            subset = [selected_elements[row] for row in np.where(column)[0]]
            input_sequence.append(subset)


        # Map the solution subsets to their corresponding indices in the input sequence
        label_sequence = [subset for idx, subset in enumerate(input_sequence) if idx in solution_indices]
        sequences.append((input_sequence, label_sequence))

        print("input sequence:", input_sequence)
        print("label sequence:", label_sequence)

    return sequences

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
num_matrices = 15
max_universe_size = 6
sequences = generate_setcover_sequences2(num_matrices, max_universe_size)

# Print the generated sequences
for i, (input_sequence, label_sequence) in enumerate(sequences):
    print(f"Sequence {i + 1} - Input:")
    print(input_sequence)
    print("Label:")
    print(label_sequence)
    print()
