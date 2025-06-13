import numpy as np
from scipy.optimize import linear_sum_assignment

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
    """Find permutation of columns that maximizes bit-wise correlation."""
    C = correlation_matrix(L1, L2)
    # Convert to cost matrix by subtracting from max (since Hungarian minimizes)
    max_val = np.max(C)
    cost_matrix = max_val - C
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind  # This gives us the new index mapping

def apply_permutation(vecs, perm):
    return [[vec[i] for i in perm] for vec in vecs]

def apply_flips(vecs, flips):
    return [[bit ^ flips[i] for i, bit in enumerate(vec)] for vec in vecs]

def hamming_distance(v1, v2):
    return sum(b1 != b2 for b1, b2 in zip(v1, v2))

def total_cost_with_matching(L1, L2):
    L1 = np.array(L1)
    L2 = np.array(L2)
    n = len(L1)

    cost_matrix = np.array([[hamming_distance(v1, v2) for v2 in L2] for v1 in L1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute Hamming distances of matched pairs
    match_pairs = [(i, col_ind[i], cost_matrix[i][col_ind[i]]) for i in range(n)]

    # Sort by Hamming distance (ascending)
    match_pairs.sort(key=lambda x: x[2])  # x[2] is the distance

    # Extract the sorted order for L1
    sorted_indices = [i for i, _, _ in match_pairs]

    # print(sorted_indices)
    L1_sorted = L1[sorted_indices].tolist()
    return L1_sorted

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

def inverse_perm(perm):
    inv = [0]*len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

def optimize_with_correlation_init(L1, L2):
    # Step 1: Correlation-based permutation
    perm = best_column_permutation(L1, L2)
    permuted_L1 = apply_permutation(L1, perm)
    inv_perm = inverse_perm(perm)

    # Step 2: Match vectors and compute optimal bit flips
    L1_sorted= total_cost_with_matching(permuted_L1, L2)
    restored_L1 = apply_permutation(L1_sorted, perm)
    restored_L2 = apply_permutation(L2, perm)


    # flips = optimal_bit_flips(permuted_L1, L2, assignment)
    # flipped_L1 = apply_flips(permuted_L1, flips)

    # # Step 3: Final matching and cost
    # cost, final_assignment = total_cost_with_matching(flipped_L1, L2)

    return restored_L1, restored_L2

# Example usage
L1 = [
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
]

L2 = [
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]

restored_L1, restored_L2 = optimize_with_correlation_init(L1, L2)
print('Restored L1', restored_L1)
print('Restored L2', restored_L2)

# print(f"Correlation-based permutation: {perm}")
# print(f"Bit flips per index: {flips}")
# print(f"Minimum total bit flips: {cost}")
# print(f"Vector matching (L1 idx → L2 idx): {match}")
