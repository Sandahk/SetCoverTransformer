import os
import io
import re
import sys
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.src.optim import get_optimizer
from SymbolicMathematics.src.utils import to_cuda
from logging import getLogger
import os
import io
import re
import sys
import math
import itertools
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.integrals.risch import NonElementaryIntegral
from sympy.calculus.util import AccumBounds

from itertools import combinations


def convert_tensor_to_token_sequence_old(self, tensor):
    # The input is a tensor. the output is a list!
    transposed_tensor = tensor.T
    return [[self.env.id2word[idx] for idx in row[1:]] for row in transposed_tensor.cpu().numpy()]


def convert_tensor_to_token_sequence(self, tensor):
    # The input is a tensor. the output is a list!
    transposed_tensor = tensor.T
    sequences = [[self.env.id2word[idx] for idx in row[1:]] for row in transposed_tensor.cpu().numpy()]

    # Flatten the list of lists into a single list
    # concatenated_sequence = [token for sequence in sequences for token in sequence]

    return sequences


def convert_pred_output_to_tokens(self, tensor):
    # the output is a sequence of tokens
    pred_output_list = tensor.cpu().numpy().tolist()
    tokens = [self.env.id2word[idx] for idx in pred_output_list]
    return tokens


def convert_token_sequence_to_list_of_lists_old(self, token_sequence):
    list_of_lists = []  # This will store the final list of lists
    current_list = []  # This will store the current list being constructed
    depth = 0  # Variable to keep track of the depth of nesting

    for token_list in token_sequence:  # Iterate over each list of tokens in the token sequence
        inner_list = []  # This will store the current inner list being constructed
        for token in token_list:  # Iterate over each token in the current token list
            if token == '[':  # If an opening bracket is encountered
                if depth > 0:
                    inner_list = []  # Start a new inner list
                depth += 1  # Increment the depth of nesting
            elif token == ']':  # If a closing bracket is encountered
                depth -= 1  # Decrement the depth of nesting
                if depth == 0:
                    if inner_list:  # Check if the inner list is not empty
                        current_list.append(inner_list)  # Append the inner list to the current list
                    list_of_lists.append(current_list)  # Append the constructed list to the final result
                    current_list = []  # Reset the current list
                else:
                    current_list.append(inner_list)  # Append the inner list to the current list
                    inner_list = []  # Start a new inner list

            elif token != '[' and token != ']' and token != '</s>' and token != ',' and token != '<s>':  # If the token is not an opening or closing bracket
                inner_list.append(token)  # Add the token to the inner list

    return list_of_lists  # Return the final list of lists


# Problem: for each row we had separate sequences. [[][][]] not as tokens


def convert_token_sequence_to_list_of_lists(self, token_sequence):
    list_of_lists = []  # This will store the final list of lists
    current_list = []  # This will store the current list being constructed
    depth = 0  # Variable to keep track of the depth of nesting
    inner_list = []  # Initialize inner_list

    if len(token_sequence) == 0:
        return []

    for token in token_sequence:  # Iterate over each token in the token sequence
        if token == '[':  # If an opening bracket is encountered
            if depth > 0:
                inner_list = []  # Start a new inner list
            depth += 1  # Increment the depth of nesting
        elif token == ']':  # If a closing bracket is encountered
            depth -= 1  # Decrement the depth of nesting
            if depth == 0:
                if inner_list:  # Check if the inner list is not empty
                    current_list.append(inner_list)  # Append the inner list to the current list
                list_of_lists.append(current_list)  # Append the constructed list to the final result
                current_list = []  # Reset the current list
            else:
                current_list.append(inner_list)  # Append the inner list to the current list
                inner_list = []  # Start a new inner list
        elif token.isdigit():  # If the token is an integer
            inner_list.append(int(token))  # Convert token to integer and add to the inner list

    if len(list_of_lists) == 0:
        return []
    return list_of_lists[0]  # Return the final list of lists


def validate_set_cover_solution(self, problem_set, solution_set, true_output):
    self.valid_solution_count = 0
    self.error_counts = {
        "Wrong Format": 0,
        "Subset not in problem set": 0,
        "Solution not covering problem set": 0,
        "Solution is not optimal": 0
    }
    self.coverage_percentage = 0

    # Check if the solution set is empty
    if not solution_set:
        self.error_counts["Wrong Format"] += 1
        return self.valid_solution_count, self.error_counts, self.coverage_percentage


    # Check if any subset in the solution set is empty
    for j, subset in enumerate(solution_set):
        if not subset:
            self.error_counts["Wrong Format"] += 1
            break

    # Check if all subsets in the solution set exist in the problem set
        if subset not in problem_set:
            self.error_counts["Subset not in problem set"] += 1
            break

    else:
        # Check if the solution set covers the entire problem set
        covered_tokens = set().union(*solution_set)
        problem_tokens = set(token for sublist in problem_set for token in sublist)
        if covered_tokens != problem_tokens:
            self.coverage_percentage = round(len(covered_tokens) / len(problem_tokens) * 100, 2)
            self.error_counts["Solution not covering problem set"] += 1

        else:
            if len(solution_set) > len(true_output):
                self.error_counts["Solution is not optimal"] += 1
            else:
                self.valid_solution_count += 1

    return self.valid_solution_count, self.error_counts, self.coverage_percentage


def validate_set_cover_solution_validation(self, problem_set, solution_set, true_output):
    self.valid_solution_count_validation = 0
    self.error_counts_validation = {
        "Wrong Format": 0,
        "Subset not in problem set": 0,
        "Solution not covering problem set": 0,
        "Solution is not optimal": 0
    }
    self.coverage_percentage_validation = 0

    # Check if the solution set is empty
    if not solution_set:
        self.error_counts_validation["Wrong Format"] += 1
        return self.valid_solution_count_validation, self.error_counts_validation, self.coverage_percentage_validation

    # Check if any subset in the solution set is empty
    for j, subset in enumerate(solution_set):
        if not subset:
            self.error_counts_validation["Wrong Format"] += 1
            return self.valid_solution_count_validation, self.error_counts_validation, self.coverage_percentage_validation

        # Check if all subsets in the solution set exist in the problem set
        if subset not in problem_set:
            self.error_counts_validation["Subset not in problem set"] += 1
            return self.valid_solution_count_validation, self.error_counts_validation, self.coverage_percentage_validation


    # Check if the solution set covers the entire problem set
    covered_tokens = set().union(*solution_set)
    problem_tokens = set(token for sublist in problem_set for token in sublist)
    if covered_tokens != problem_tokens:
        self.coverage_percentage_validation = round(len(covered_tokens) / len(problem_tokens) * 100, 2)
        self.error_counts_validation["Solution not covering problem set"] += 1
        return self.valid_solution_count_validation, self.error_counts_validation, self.coverage_percentage_validation


    if len(solution_set) > len(true_output):
        self.error_counts_validation["Solution is not optimal"] += 1
    else:
        self.valid_solution_count_validation += 1

    return self.valid_solution_count_validation, self.error_counts_validation, self.coverage_percentage_validation


# def extract_lists(char_list)
def list_to_tokens(lst):
    return [str(c) for c in str(lst) if str(c) != ' ']
