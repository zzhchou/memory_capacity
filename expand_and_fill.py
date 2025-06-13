import itertools as it
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from collections import Counter

def pad_combs(comb_list, first_N, second_N):
    padded_comb_list = []
    for comb in comb_list:
        padded_comb_list.append(np.pad(comb, (0, second_N-first_N), 'constant'))
    return padded_comb_list

def build_weight_matrix(patt_list):
    N = len(patt_list[0])
    weight_matrix = np.zeros((N,N), dtype=int)
    for patt in patt_list:
        ones_idx = [i for i, val in enumerate(patt) if val == 1]
        for i, j in it.product(ones_idx, repeat=2):
            weight_matrix[i][j] = 1
    return weight_matrix

def is_valid(patt_list, threshold):
    weight_matrix = build_weight_matrix(patt_list)

    for patt in patt_list:
        patt_arr = np.array(patt)
        product_vec = weight_matrix @ patt
        thresholded = (product_vec >= threshold).astype(int)
        if not np.array_equal(thresholded, patt):
            return False
    return True

def array_in_list(array, array_list):
    return any(np.array_equal(array, a) for a in array_list)

def fill_random(patt_list, comb_list, threshold):
    random.shuffle(comb_list)
    for i in range(len(comb_list)-1, -1, -1):
        random_comb = comb_list[i]
        if not array_in_list(random_comb, patt_list):
            patt_list.append(random_comb)
            del comb_list[i]    
            if(is_valid(patt_list, threshold)):
                return patt_list, comb_list, True
            else:
                patt_list.pop()
    return patt_list, comb_list, False

def generate_random_patterns(patt_list, comb_list, threshold):
    valid_flag = True
    while valid_flag == True:
        final_patt_list, remaining_comb_list, valid_flag = fill_random(patt_list, comb_list, T)
    return np.array(final_patt_list)

def generate_all_combinations(N, S):
    elements = np.arange(N)
    combinations = list(it.combinations(elements, S))
    comb_arr = np.array(combinations)
    comb_list = []
    for indexes in comb_arr:
        id = np.zeros(N, dtype=np.int32)
        for index in indexes:
            id[index] = 1
        comb_list.append(id)
    return comb_list

def generate_submatrix(S):
    # perm = np.random.permutation(S)
    perm = np.arange(S)
    matrix = np.ones((S,S), dtype=int)
    for i in range(S):
        matrix[i, perm[i]] = 0
    return matrix

def wrap_index(i, shift, S):
    if (i + shift) > S-1:
        new_index = i+shift-S
    elif (i + shift) < 0:
        new_index = i+shift+S
    else:
        new_index = i+shift
    return new_index

def flip_index(i, j, S):
    if S[i][j] == 0:
        S[i][j] = 1
    elif S[i][j] == 1:
        S[i][j] = 0
    return S
    
def generate_symmetric_submatrix(S, T):
    matrix = np.ones((S,S), dtype=int)
    max_shift = (S-T)//2

    if (S-T) % 2 == 0:
        for i in range(S):
            for j in range(-max_shift, max_shift+1):
                wrapped_j = wrap_index(S-i-1, j, S)
                # print(i,wrapped_j)
                if S-i-1 != wrapped_j:
                    matrix[i][wrapped_j] = 0
        matrix = flip_index(0,0,matrix)
        matrix = flip_index(S-1, S-1, matrix)
        matrix = flip_index(0, S-1, matrix)
        matrix - flip_index(S-1, 0, matrix)
    else:
        for i in range(S):
            for j in range(-max_shift, max_shift+1):
                wrapped_j = wrap_index(S-i-1, j, S)
                matrix[i][wrapped_j] = 0
    
    return matrix

def generate_remainder_matrix(final_matrix, S, R, T):
    N = len(final_matrix)
    B = N//S
    F = B*S-1
    matrix = np.ones((R,R), dtype=int)
    max_shift = 0

    # print(F)
    for i in range(N):
        for j in range(N):
            if i > F and j > F:
                if i == j:
                    final_matrix[i][j] = 1
            elif i > F:
                if j < S:
                    if j%S != S-1:
                        final_matrix[i][j] = 1
                else:
                    if j%S != 0:
                        final_matrix[i][j] = 1                    
            elif j > F:
                if i < S:
                    if i%S != S-1:
                        final_matrix[i][j] = 1
                else:
                    if i%S != 0:
                        final_matrix[i][j] = 1 

    return final_matrix

def invert_matrix(matrix):
    S = len(matrix)
    new_matrix = np.zeros((S, S), dtype=int)
    for i in range(S):
        for j in range(S):
            if matrix[i][j] == 1:
                new_matrix[i][S-j-1] = 1
            else:
                new_matrix[i][S-j-1] = 0

    # print(new_matrix)
    return new_matrix

def repeat_matrix(N, S, T):
    B = N//S
    R = N%S

    final_matrix = np.zeros((N, N), dtype=int)
    # sub_matrix = generate_submatrix(S)
    sub_matrix = generate_symmetric_submatrix(S, T)
    inverted_sub_matrix = invert_matrix(sub_matrix)
    for i in range(B):
        for j in range(B):
            # If we are at a diagonal block, fill it with ones
            if i == j:
                final_matrix[i * S:(i + 1) * S, j * S:(j + 1) * S] = np.ones((S, S), dtype=int)
            elif i == 0 or j == 0:
                # Otherwise, fill the block with the permutation matrix
                final_matrix[i * S:(i + 1) * S, j * S:(j + 1) * S] = sub_matrix
            else:
                final_matrix[i * S:(i + 1) * S, j * S:(j + 1) * S] = inverted_sub_matrix

    remainder_matrix = generate_remainder_matrix(final_matrix, S, R, T)


    return remainder_matrix

def validate_pattern_error(pattern, matrix, threshold):
    output_sums = np.dot(pattern,matrix)
    # print(output_sums)
    output_pattern = [1 if sum > threshold else 0 for sum in output_sums]
    if pattern == output_pattern:
        return 1
    else:
        # print(pattern)
        # print(output_sums)
        # print(output_pattern)
        return 0

def subsample(selected_indices, matrix, S, T):
    N = len(matrix)
    sliced_matrix = matrix[selected_indices]

    sliced_matrix_sums = np.sum(sliced_matrix, axis=0)
    threshold = len(selected_indices)
    next_indices = np.where(sliced_matrix_sums >= threshold)[0]
    new_selected_indices = [selected_indices+[idx] for idx in next_indices if idx > selected_indices[-1]]
    patterns = set()

    if new_selected_indices == []:
        return patterns
    if len(new_selected_indices[0]) == S:
        for idx in new_selected_indices:
            pattern = [0] * N
            for i in idx:
                pattern[i] = 1
            validate_flag = validate_pattern_error(pattern, matrix, T)
            if validate_flag == 1:
                patterns.add(tuple(pattern))
            # else:
            #     print(pattern)
        # print(patterns)
        return patterns
    else:
        for idx in new_selected_indices:
            sampled_patterns = subsample(idx, matrix, S, T)
            patterns.update(sampled_patterns)
    return patterns

def generate_subsampled_patterns(N, S, threshold):
    T = threshold - 1
    test_matrix = generate_symmetric_submatrix(S, T)
    # print(test_matrix)

    binary_matrix = repeat_matrix(N, S, T)

    # test_pattern = [1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0, 0]
    # test_flag = validate_pattern_error(test_pattern, binary_matrix, T)
    # print ('Test Flag:', test_flag)
    # print(binary_matrix)

    candidates = np.arange(N)
    candidates = [[_] for _ in candidates]

    pattern_list = set()
    for candidate in candidates:
        pattern_list.update(subsample(candidate, binary_matrix, S, T))

    pattern_list = list(pattern_list)
    return pattern_list

first_N = 16
second_N = 20

S = 4
T = 4

# CASE 1
first_comb_list = generate_all_combinations(first_N, S)
padded_comb_list = pad_combs(first_comb_list, first_N, second_N)
first_patt_list = []
partial_patt_list = generate_random_patterns(first_patt_list, padded_comb_list, T)
capacity_random_1 = len(partial_patt_list)

second_comb_list = generate_all_combinations(second_N, S)
second_patt_list = partial_patt_list.tolist()
final_patt_list = generate_random_patterns(second_patt_list, second_comb_list, T)
capacity_random_2 = len(final_patt_list)

# CASE 2
subsampled_patt_list_1 = generate_subsampled_patterns(first_N, S, T)
capacity_subsampled_1 = len(subsampled_patt_list_1)
subsampled_patt_list_2 = generate_subsampled_patterns(second_N, S, T)
capacity_subsampled_2 = len(subsampled_patt_list_2)

# CASE 3
subsampled_patt_list_1 = generate_subsampled_patterns(first_N, S, T)
capacity_mixed_1 = len(subsampled_patt_list_1)
padded_patt_list = pad_combs(subsampled_patt_list_1, first_N, second_N)
second_comb_list = generate_all_combinations(second_N, S)
# second_patt_list = subsampled_patt_list_1
final_patt_list = generate_random_patterns(padded_patt_list, second_comb_list, T)
capacity_mixed_2 = len(final_patt_list)

# test_pattern_set = [   [1, 1, 1, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 1, 1, 1, 0, 0, 0],
#                        [1, 1, 0, 1, 0, 0, 0, 0, 0],
#                        [1, 1, 0, 0, 0, 0, 1, 0, 0],
#                        [1, 0, 1, 0, 1, 0, 0, 0, 0],
#                        [1, 0, 1, 0, 0, 0, 0, 1, 0],
#                        [0, 1, 1, 0, 0, 1, 0, 0, 0],
#                        [0, 1, 1, 0, 0, 0, 0, 0, 1],
#                        [1, 0, 0, 1, 1, 0, 0, 0, 0],
#                        [1, 0, 0, 1, 0, 0, 0, 1, 0],
#                        [1, 0, 0, 0, 1, 0, 1, 0, 0],
#                        [1, 0, 0, 0, 0, 0, 1, 1, 0],
#                        [0, 1, 0, 1, 0, 1, 0, 0, 0],
#                        [0, 1, 0, 1, 0, 0, 0, 0, 1],
#                        [0, 1, 0, 0, 0, 1, 1, 0, 0],
#                        [0, 1, 0, 0, 0, 0, 1, 0, 1],
#                        [0, 0, 1, 0, 1, 1, 0, 0, 0],
#                        [0, 0, 1, 0, 1, 0, 0, 0, 1],
#                        [0, 0, 1, 0, 0, 1, 0, 1, 0],
#                        [0, 0, 1, 0, 0, 0, 0, 1, 1],
#                        [0, 0, 0, 1, 1, 0, 0, 0, 1],
#                        [0, 0, 0, 1, 0, 1, 0, 1, 0],
#                        [0, 0, 0, 0, 1, 1, 1, 0, 0],
#                        [0, 0, 0, 0, 0, 1, 1, 1, 0],
#                        [0, 0, 0, 0, 1, 0, 1, 0, 1],
#                        [0, 0, 0, 1, 0, 0, 0, 1, 1],
#                        [0, 0, 0, 0, 0, 0, 1, 1, 1]
#                     ]

# valid_flag = is_valid(test_pattern_set, 3)