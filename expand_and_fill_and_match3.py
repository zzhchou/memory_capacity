import itertools as it
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from collections import Counter
from scipy.optimize import linear_sum_assignment

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

def validate_pattern_error_offset(pattern, matrix, threshold):
    output_sums = np.dot(pattern,matrix)
    # print(output_sums)
    output_pattern = [1 if sum > threshold-1 else 0 for sum in output_sums]
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

def hamming_distance(a, b):
    return np.sum(a != b)

def best_column_permutation(optimal, suboptimal):
    """
    Permute columns of optimal to minimize total Hamming distance to suboptimal.
    """
    n_cols = optimal.shape[1]
    best_perm = None
    min_total_distance = float('inf')
    
    for perm in it.permutations(range(n_cols)):
        permuted = optimal[:, perm]
        total_distance = sum(
            min(hamming_distance(opt_row, sub_row) for sub_row in suboptimal)
            for opt_row in permuted
        )
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_perm = perm

    print('Done')
    return optimal[:, best_perm]

def frequency_based_column_matching(optimal, suboptimal):
    """
    Match columns of optimal to suboptimal based on 1s frequency.
    """
    opt_freq = np.sum(optimal, axis=0)
    sub_freq = np.sum(suboptimal, axis=0)

    opt_indices = np.argsort(-opt_freq)  # sort descending
    sub_indices = np.argsort(-sub_freq)

    # Create a column mapping from optimal to suboptimal
    col_map = np.zeros_like(opt_indices)
    for i, idx in enumerate(opt_indices):
        col_map[idx] = sub_indices[i]

    # Apply column permutation
    inverse_map = np.argsort(col_map)
    return optimal[:, inverse_map]


def hungarian_column_matching(optimal, suboptimal):
    """
    Match columns of optimal to suboptimal using Hungarian algorithm.
    Columns are matched based on frequency of 1s (normalized).
    """
    n_opt_cols = optimal.shape[1]
    n_sub_cols = suboptimal.shape[1]
    
    # Create cost matrix between each optimal and suboptimal column
    cost_matrix = np.zeros((n_opt_cols, n_sub_cols))
    
    opt_freq = np.mean(optimal, axis=0)  # shape: (n_opt_cols,)
    sub_freq = np.mean(suboptimal, axis=0)  # shape: (n_sub_cols,)

    for i in range(n_opt_cols):
        for j in range(n_sub_cols):
            cost_matrix[i, j] = abs(opt_freq[i] - sub_freq[j])

    # Solve assignment (for square cost matrix, use min(n, m) x min(n, m))
    n = min(n_opt_cols, n_sub_cols)
    row_ind, col_ind = linear_sum_assignment(cost_matrix[:n, :n])

    # Create column permutation for optimal
    matched_cols = col_ind
    permuted_optimal = optimal[:, matched_cols]

    return permuted_optimal

def sort_by_closest_match(permuted_optimal, suboptimal):
    """
    Sort suboptimal by closest match to any row in permuted optimal.
    """
    distances = []
    for sub in suboptimal:
        min_dist = min(hamming_distance(sub, opt) for opt in permuted_optimal)
        distances.append(min_dist)
    
    sorted_indices = np.argsort(distances)[::-1]  # worst to best
    return suboptimal[sorted_indices]

def concat_unique_arrays(list1, list2):
    seen = set()
    result = []
    expected_shape = list1[0].shape  # assume all should match
    for arr in np.concatenate((list1,list2), axis=0):
        if arr.shape != expected_shape:
            raise ValueError(f"Inconsistent shape: expected {expected_shape}, got {arr.shape}")
        key = tuple(arr.flatten())  # flatten ensures 2D arrays hashable
        if key not in seen:
            seen.add(key)
            result.append(arr)
    return result

def compatibility_test(optimal, suboptimal, deleted_patts, threshold):
    # print('New Test')
    num_passes = 0
    combined_patts = concat_unique_arrays(suboptimal, optimal)
    # combined_array = suboptimal
    combined_array = combined_patts
    # print(len(suboptimal), len(optimal), len(combined_patts))
    for patt in combined_patts:
        if sum(patt) == 0:
            pass_flag = 0
            continue
        new_combined_array = np.concatenate((combined_array, np.asarray([patt])), axis=0)
        weight_matrix = build_weight_matrix(new_combined_array)        
        # print ('compatibility_test patt', patt)
        patt_list = patt.tolist()
        pass_flag = validate_pattern_error_offset(patt_list, weight_matrix, threshold)
        # print(patt_list, pass_flag)
        if pass_flag == 1:
            num_passes += 1
            combined_array = new_combined_array

    # Add back suboptimal patterns if no optimal patterns can be added
    # for patt2 in deleted_patts:
    #     new_combined_array = np.concatenate((combined_array, np.asarray([patt2])), axis=0)
    #     weight_matrix = build_weight_matrix(combined_array)        
    #     # print ('compatibility_test patt', patt)
    #     patt_list = patt.tolist()
    #     pass_flag = validate_pattern_error_offset(patt_list, weight_matrix, threshold)
    #     if pass_flag == 1:
    #         num_passes += 1
    #         combined_array = new_combined_array

    return num_passes

def iterative_filter2(optimal, suboptimal, hamming_distances, threshold):
    """
    Iteratively remove worst suboptimal patterns and perform compatibility test.
    """
    suboptimal_copy = copy.deepcopy(suboptimal)
    initial_len = len(suboptimal_copy)
    maxed_flag = 0
    deleted_count = 0
    deleted_count_list = []
    passed_count_list = []
    deleted_patts = []
    hamming_distance_list = []
    modified_count = 0
    modified_list = []
    must_delete_list = []

    # print('Iterative filter optimal', len(optimal))
    original_sums = np.sum(suboptimal_copy, axis=1)
    original_copy = [_ for _ in original_sums if _ != 0]
    original_passed = len(original_copy)
    original_zeros = len(optimal) - original_passed

    for i in range(len(suboptimal_copy) - original_zeros):
        while not np.array_equal(suboptimal_copy[i], optimal[i]):
            passed = compatibility_test(optimal, suboptimal_copy, deleted_patts, threshold)
            if passed <= original_passed:
                passed = original_passed
            # print(deleted_count, passed)
            if passed == len(optimal) and maxed_flag == 0:
                print(modified_count)
                for k in range(len(suboptimal_copy)):
                    print(suboptimal_copy[k], optimal[k])
                maxed_flag = 1
            

            passed_count_list.append(passed)


            diff_indices = np.where(suboptimal_copy[i] != optimal[i])[0]
            if diff_indices.size > 0:
                flip_idx = diff_indices[0]
                hamming_preflip = hamming_distance(suboptimal_copy[i], optimal[i])     
                hamming_distance_list.append(hamming_preflip)           
                suboptimal_copy[i][flip_idx] = optimal[i][flip_idx]
                # print(suboptimal_copy[i], optimal[i], len(suboptimal_copy), len(optimal))
                # yield suboptimal.copy()
                modified_count += 1
                modified_list.append(modified_count)

                hamming_sum = 0
                must_delete = 0
                for j in range(len(suboptimal_copy) - original_zeros):
                    if not np.array_equal(suboptimal_copy[j], optimal[j]):
                        must_delete += 1
                    hamming_sum += hamming_distance(suboptimal_copy[j], optimal[j])
                # hamming_distance_list.append(hamming_sum)
                must_delete_list.append(must_delete)


            # deleted_patts.append(suboptimal_copy[-1])
            # HD_index1 = len(suboptimal_copy) - 1
            # print('HD', hamming_absolute, hamming_relative)
            # suboptimal_copy = suboptimal_copy[:-1]  # delete worst (last) pattern
            # deleted_count += 1
            # deleted_count_list.append(deleted_count)

    # print(len(modified_list), len(passed_count_list), len(hamming_distance_list))
    # passed_count = np.sum(compatibility_test(optimal, suboptimal))
    # return modified_list, passed_count_list, hamming_distance_list
    return modified_list, passed_count_list, hamming_distance_list, must_delete_list

def correlation_matrix(L1, L2):
    """Compute bit-wise agreement counts between L1 and L2 columns."""
    L1 = np.array(L1)
    L2 = np.array(L2)
    d = L1.shape[1]
    C = np.zeros((d, d), dtype=int)

    for i in range(d):
        for j in range(d):
            match = np.sum(L1[:, i] == L2[:, j])  # Count agreements across vectors
            C[i, j] = match
    return C

def best_column_permutation(L1, L2):
    """Find the best column permutation to align L1 to L2 based on matching bits.
    Handles cases where L1 and L2 have different numbers of rows (vectors)."""
    
    L1 = np.array(L1)
    L2 = np.array(L2)

    n = min(L1.shape[0], L2.shape[0])  # compare only first n rows
    d = L1.shape[1]

    if L2.shape[1] != d:
        raise ValueError("L1 and L2 must have the same number of columns (features).")

    C = np.zeros((d, d), dtype=int)

    # Build similarity matrix based on overlapping vectors
    for i in range(d):
        for j in range(d):
            match = np.sum(L1[:n, i] == L2[:n, j])
            C[i, j] = match

    # Use Hungarian algorithm to find max match permutation
    cost_matrix = C.max() - C
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return col_ind.tolist()  # This is the column permutation to apply to L1

def apply_permutation(vecs, perm):
    return [[vec[i] for i in perm] for vec in vecs]

def apply_flips(vecs, flips):
    return [[bit ^ flips[i] for i, bit in enumerate(vec)] for vec in vecs]

def hamming_distance2(v1, v2):
    return sum(b1 != b2 for b1, b2 in zip(v1, v2))

def total_cost_with_matching(L1, L2):
    L1 = np.array(L1)
    L2 = np.array(L2)

    n1 = len(L1)
    n2 = len(L2)

    if L1.shape[1] != L2.shape[1]:
        raise ValueError("L1 and L2 must have the same number of columns.")

    # Pad L1 with zero vectors if it's shorter than L2
    if n1 < n2:
        padding = np.zeros((n2 - n1, L1.shape[1]), dtype=int)
        L1 = np.vstack([L1, padding])

    # Compute full cost matrix
    cost_matrix = np.array([[hamming_distance2(v1, v2) for v2 in L2] for v1 in L1])

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get matched pairs (only the first n2 since we care about L2's full length)
    match_pairs = [(i, j, cost_matrix[i][j]) for i, j in zip(row_ind, col_ind) if j < n2]

    # Sort matched pairs by Hamming distance
    match_pairs.sort(key=lambda x: x[2])

    # Get sorted L1 and L2 according to matching
    sorted_indices_L1 = [i for i, _, _ in match_pairs]
    sorted_indices_L2 = [j for _, j, _ in match_pairs]
    sorted_hamming_distances = [k for _, _, k in match_pairs]

    L1_sorted = L1[sorted_indices_L1].tolist()
    L2_sorted = L2[sorted_indices_L2].tolist()


    # Testing effect of different orders
    L1_sorted.reverse()
    sorted_hamming_distances.reverse()

    return L1_sorted, L2_sorted, sorted_hamming_distances

def optimal_bit_flips(L1, L2, assignment):
    d = len(L1[0])
    flips = [0] * d
    for i in range(d):
        flip_cost = 0
        no_flip_cost = 0
        for l1_idx, l2_idx in assignment:
            if L1[l1_idx][i] != L2[l2_idx][i]:
                no_flip_cost += 1
            else:
                flip_cost += 1
        if flip_cost < no_flip_cost:
            flips[i] = 1
    return flips

# def optimize_with_correlation_init(L1, L2):
#     # Step 1: Correlation-based permutation
#     perm = best_column_permutation(L1, L2)
#     permuted_L1 = apply_permutation(L1, perm)

#     # Step 2: Match vectors and compute optimal bit flips
#     _, assignment = total_cost_with_matching(permuted_L1, L2)
#     flips = optimal_bit_flips(permuted_L1, L2, assignment)
#     flipped_L1 = apply_flips(permuted_L1, flips)

#     # Step 3: Final matching and cost
#     cost, final_assignment = total_cost_with_matching(flipped_L1, L2)

#     return perm, flips, cost, final_assignment

def inverse_perm(perm):
    inv = [0]*len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

def sort_and_restore_lists(L1, L2):
    # Step 1: Correlation-based permutation
    perm = best_column_permutation(L1, L2)
    permuted_L1 = apply_permutation(L1, perm)
    inv_perm = inverse_perm(perm)
    # print(len(L2))
    # Step 2: Match vectors and compute optimal bit flips
    L1_sorted, L2_sorted, sorted_HD = total_cost_with_matching(permuted_L1, L2)
    # print(len(L2_sorted))
    restored_L1 = apply_permutation(L1_sorted, inv_perm)
    restored_L2 = apply_permutation(L2_sorted, inv_perm)

    print(len(restored_L2))
    return restored_L1, restored_L2, sorted_HD

def iterative_replacement(suboptimal_patt_list, optimal_patt_list, threshold):
    L1, L2, HD = sort_and_restore_lists(suboptimal_patt_list, optimal_patt_list)
    # print('Permuted Optimal', len(L2), L2)
    # print('Sorted Suboptimal', len(L1), L1)

    L1_array = np.asarray(L1)
    L2_array = np.asarray(L2)
    HD_array = np.asarray(HD)

    num_flipped, num_passed, hamming_distance_removed, patts_removed = iterative_filter2(L2_array, L1_array, HD_array, threshold)

    return num_flipped, num_passed, hamming_distance_removed, patts_removed



N = 16
# second_N = 20

S = 4
T = 4

# # CASE 1
comb_list = generate_all_combinations(N, S)
# print('comb_list', len(comb_list))
first_patt_list = []
suboptimal_patt_list = generate_random_patterns(first_patt_list, comb_list, T)
optimal_patt_list = generate_subsampled_patterns(N, S, T)

print('optimal_patt_list', len(optimal_patt_list))
print('suboptimal_patt_list', len(suboptimal_patt_list))
bits_flipped, removed_capacity_list, hamming_distance_removed, patts_removed = iterative_replacement(suboptimal_patt_list, optimal_patt_list, T)

removed_capacity_normalized = [_*100/max(removed_capacity_list) for _ in removed_capacity_list]
hamming_distance_normalized = [_/max(hamming_distance_removed) for _ in hamming_distance_removed]

plt.figure()
plt.plot(bits_flipped, removed_capacity_normalized)
plt.xlabel('Bits Flipped')
plt.ylabel('% Capacity')
plt.show()

plt.figure()
plt.plot(bits_flipped, patts_removed)
# plt.plot(bits_flipped, hamming_distance_removed)
plt.xlabel('Bits Flipped')
plt.ylabel('Patterns Removed')
plt.savefig('Cost_Tradeoff4.eps', format='eps')
plt.show()

# # CASE 1
# first_comb_list = generate_all_combinations(first_N, S)
# padded_comb_list = pad_combs(first_comb_list, first_N, second_N)
# first_patt_list = []
# partial_patt_list = generate_random_patterns(first_patt_list, padded_comb_list, T)
# capacity_random_1 = len(partial_patt_list)

# second_comb_list = generate_all_combinations(second_N, S)
# second_patt_list = partial_patt_list.tolist()
# final_patt_list = generate_random_patterns(second_patt_list, second_comb_list, T)
# capacity_random_2 = len(final_patt_list)

# # CASE 2
# subsampled_patt_list_1 = generate_subsampled_patterns(first_N, S, T)
# capacity_subsampled_1 = len(subsampled_patt_list_1)
# subsampled_patt_list_2 = generate_subsampled_patterns(second_N, S, T)
# capacity_subsampled_2 = len(subsampled_patt_list_2)

# # CASE 3
# subsampled_patt_list_1 = generate_subsampled_patterns(first_N, S, T)
# capacity_mixed_1 = len(subsampled_patt_list_1)
# padded_patt_list = pad_combs(subsampled_patt_list_1, first_N, second_N)
# second_comb_list = generate_all_combinations(second_N, S)
# # second_patt_list = subsampled_patt_list_1
# final_patt_list = generate_random_patterns(padded_patt_list, second_comb_list, T)
# capacity_mixed_2 = len(final_patt_list)

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