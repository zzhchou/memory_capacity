import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from collections import Counter

# np.random.seed(5)
np.random.seed(12)

def validate_pattern_set(pattern_set, total_num_patterns):
    NCELL = len(pattern_set[0])
    NPATT = len(pattern_set)
    SPATT = sum(pattern_set[0])
    pre_threshold_out = np.zeros((NCELL, NPATT))

    p = np.array(pattern_set).T
    w = np.zeros((NCELL, NCELL))

    for pattern in pattern_set:
        w = w+np.outer(pattern, pattern)

    ACTIVATION_THRESHOLD = sum(pattern_set[0]) - 2
    for IP_ID in range(len(pattern_set)):
        average_percent_error, average_crosspattern_activation, max_crosspattern_activation, min_crosspattern_activation, cpa_counts, max_error, min_error, input_pattern, output_pattern, pre_threshold = calculate_specific_error(IP_ID, ACTIVATION_THRESHOLD, w, p)
    
        pre_threshold_out[:, IP_ID] = pre_threshold

    # print(pre_threshold)
    w_jdx = pre_threshold_out
    plt.rcParams['font.size'] = 24
    ax = plt.figure(figsize=(12,8), dpi=300).gca()
    im = plt.imshow(w_jdx, aspect='auto', interpolation='nearest', cmap='plasma', vmin=0, vmax=SPATT)
    plt.xlabel('Input Pattern ID #', labelpad=10)
    plt.ylabel('Output Cell ID #', labelpad=10)
    # plt.title('CPA Counts')
    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

    # ax.set_xticks(np.arange(-.5, w_jdx.shape[0]+0.5, 1), minor=False)
    # ax.set_yticks(np.arange(-.5, w_jdx.shape[0]+0.5, 5), minor=False)
    # ax.grid(which='major', color='black', linewidth=1)
    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)

    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.xlim(-0.5, total_num_patterns+.5)
   
    cbar = plt.colorbar(im, ax=ax)
    cbar_ticks = range(SPATT+1)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.set_label('CPA Values', rotation=270, labelpad=30)
    plt.savefig("cpa_heatmap_{}.svg".format(total_num_patterns))

    plt.show()

    w_jdx2 = np.clip(w, 0, 1)

    plt.rcParams['font.size'] = 24
    # plt.rcParams['xtick.major.pad'] = '8'
    # plt.rcParams['ytick.major.pad'] = '8'

    ax = plt.figure(figsize=(8,8), dpi=300).gca()
    im = plt.imshow(w_jdx2, aspect='auto', interpolation='nearest', cmap='binary', vmin=-0.2, vmax=1)
    plt.xlabel('Output Cell ID #', labelpad=10)
    plt.ylabel('Input Cell ID #', labelpad=10)
    # plt.title('Connection Matrix',)
    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True))
    

    # ax.set_xticks(np.arange(-.5, w_jdx2.shape[1]+0.5, 5), minor=False)
    # ax.set_yticks(np.arange(-.5, w_jdx2.shape[0]+0.5, 5), minor=False)
    # ax.grid(which='major', color='black', linewidth=1)
    ax.set_xticks(np.arange(w_jdx2.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(w_jdx2.shape[0]+1)-.5, minor=True)

    # plt.xlim(-0.5, w_jdx2.shape[1]-.5)
    # plt.ylim(-0.5, w_jdx2.shape[0]-.5)


    # plt.colorbar(im, ax=ax)

    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.savefig("connectivity_matrix_{}.svg".format(total_num_patterns))
    plt.show()

    return

def calculate_specific_error(PATT_ID, ACTIVATION_THRESHOLD, w, p):
    all_total_errors = []
    all_crosspattern_activation = []
    CUE = PATT_ID
    input_pattern = p[:,CUE]

    active_input_cells = np.where(p[:,CUE] == 1)[0]
    activation_count = np.zeros(len(w[0]), dtype=int)
    for output_cell in range(0, len(w[0])):
        for active_input in active_input_cells:
            if w[output_cell][active_input].any() == 1:
                activation_count[output_cell] = activation_count[output_cell] + 1

    # print(activation_count, ACTIVATION_THRESHOLD)
    # print("W: ", w)
    # print("P: ", p)
    output_pattern = activation_count > ACTIVATION_THRESHOLD
    # print(p[:,CUE])
    # print(output_pattern)

    # if sum(output_pattern) == 0:
    #     print("WARNING: No output activity")
    spurious_error = 0
    deleterious_error = 0
    for i in range(0, len(output_pattern)):
        if p[i][CUE].any() == 0:            # non-active cell
            all_crosspattern_activation.append(activation_count[i])                

        if output_pattern[i].any() == 1 and p[i][CUE].any() == 0:
            spurious_error = spurious_error + 1
            # all_crosspattern_activation.append(activation_count[i])                
        elif output_pattern[i].any() == 0 and p[i][CUE].any() == 1:
            deleterious_error = deleterious_error + 1
    total_error = spurious_error + deleterious_error
    all_total_errors.append(total_error)
    # total_error_count += total_error
    max_error = max(all_total_errors)
    min_error = min(all_total_errors)
    
    max_cpa = max(all_crosspattern_activation)
    min_cpa = min(all_crosspattern_activation)

    cpa_counts = dict(Counter(all_crosspattern_activation))



    # if NPATT == 1:
    #     print(max_error, min_error)

    average_total_error = sum(all_total_errors)/len(all_total_errors)
    average_percent_error = average_total_error*100/len(w[0])
    if len(all_crosspattern_activation) != 0:
        average_crosspattern_activation = sum(all_crosspattern_activation)/len(all_crosspattern_activation)
    else:
        average_crosspattern_activation = 0

    return average_percent_error, average_crosspattern_activation, max_cpa, min_cpa, cpa_counts, max_error, min_error, input_pattern, output_pattern, activation_count


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

    print(F)
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
        print(pattern)
        print(output_sums)
        print(output_pattern)
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
    
N = 20
S = 4
T = S-1

test_matrix = generate_symmetric_submatrix(S,T)
# print(test_matrix)

binary_matrix = repeat_matrix(N, S, T)

# test_pattern = [1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0, 0]
# test_flag = validate_pattern_error(test_pattern, binary_matrix, T)
# print ('Test Flag:', test_flag)
print(binary_matrix)

candidates = np.arange(N)
candidates = [[_] for _ in candidates]

pattern_list = set()
for candidate in candidates:
    pattern_list.update(subsample(candidate, binary_matrix, S, T))

pattern_list = list(pattern_list)
validate_pattern_set(pattern_list, len(pattern_list))

# # print(pattern_list)
print (len(pattern_list), "Patterns Stored")

