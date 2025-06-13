import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from collections import Counter
import random

n=9
k=3
THRESHOLD_DIFF = 0
comb_list = []
completed = []
failed = []

matrix = np.zeros((n,n), dtype=np.int32)

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
    # plt.show()

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


def add(id):
    global completed
    global matrix
    global failed

    old_matrix = np.copy(matrix)
    # print("identity:", id)
    id_idx = 0
    j = 0
    for i in id:
        if i > 0:
            matrix[j] = np.array(matrix[j]) + np.array(id)
            id_idx += 1
        j += 1
    # print("after:")
    # print(matrix)
    if works(id) == 0:
        completed.append(id)
        remove_patt(id)
    else:
        failed.append(id)
        matrix = old_matrix

def hamming_distance(pattern1, pattern2):
    """Calculate the Hamming distance between two binary patterns."""
    return sum(c1 != c2 for c1, c2 in zip(pattern1, pattern2))

def remove_patt(id):
    global comb_list
    global THRESHOLD_DIFF
    global failed

    fail_list = [comb for comb in comb_list if hamming_distance(comb,id) <= THRESHOLD_DIFF*2]
    comb_list = [comb for comb in comb_list if hamming_distance(comb,id) > THRESHOLD_DIFF*2]

    failed = failed+fail_list

    return

def works(id):
    error = 0
    cnt = np.zeros(n, dtype=np.int32)
    id_idx = 0
    for i in id:
        if i > 0:
            cnt_idx = 0
            for m in matrix[id_idx]:
                if m > 0:
                    # print(id_idx, cnt_idx, i, m, cnt)
                    cnt[cnt_idx] += 1
                cnt_idx += 1
        id_idx += 1
    # print("candidate =", id)
    # print("matrix =", matrix)
    # print("count =", cnt)
    id_idx = 0
    THRESHOLD = k-THRESHOLD_DIFF
    for c in cnt:
        if c > k:
            # print("cnt[", id_idx, "] = ", c, "is larger than ", k)
            error = 1
        if c < THRESHOLD:
            if id[id_idx] > 0:
                # print("cnt[", id_idx, "] = ", c, "is less than ", k)
                error = 1
        if c >= THRESHOLD:
            if id[id_idx] == 0:
                # print("cnt[", id_idx, "] = ", c, "is equal to ", k)
                error = 1
        id_idx += 1
    return error

elements = np.arange(n)
combinations = list(it.combinations(elements, k))
comb_arr = np.array(combinations)

rand_combs = np.array(combinations)
random.shuffle(rand_combs)
rand_comb_list = []

for indexes in rand_combs:
    id = np.zeros(n, dtype=np.int32)
    for index in indexes:
        id[index] = 1
    rand_comb_list.append(id)

for indexes in comb_arr:
    id = np.zeros(n, dtype=np.int32)
    for index in indexes:
        id[index] = 1
    comb_list.append(id)

def process_level(level):
    for i in range(0, n//k):
        process_block(i, level)

def process_block(block, level):
    # candidates = pop_list(block, level)
    candidates = random_pop_list(block, level)
    # print("candidates at block index ", block, " level ", level, ":")
    # random.shuffle(candidates)
    for id in candidates:
        # print(id)
        add(id)

def random_pop_list(block, level):
    global comb_list
    global completed

    candidates = []
    candidate_idxs = []
    for id_idx in range(0, len(comb_list)):
        rand_idx = random.randint(0, len(comb_list)-1)
        
        cnt = 0
        for i in range(0, k):
            if len(completed) < 64:
                id = comb_list[rand_idx] 
            else:              
                id = comb_list[rand_idx]

            if id[block*k+i] > 0:
                cnt += 1
        if cnt == level:
            if len(completed) < 64:
                candidate_idxs.append(rand_idx)
            else:
                candidate_idxs.append(rand_idx)
            candidates.append(id)
    remaining = [item for i, item in enumerate(comb_list) if i not in candidate_idxs]
    comb_list = remaining
    return candidates

def pop_list(block, level):
    global comb_list
    candidates = []
    candidate_idxs = []
    for id_idx in range(0, len(comb_list)):
        cnt = 0
        for i in range(0, k):
            id = comb_list[id_idx]
            if id[block*k+i] > 0:
                cnt += 1
        if cnt == level:
            candidate_idxs.append(id_idx)
            candidates.append(id)
    remaining = [item for i, item in enumerate(comb_list) if i not in candidate_idxs]
    comb_list = remaining
    return candidates

for i in range(0, k):
    process_level(k-i)



print("failed pass 1: ", len(failed))
for id in failed:
    comb_list.append(id)
failed = []
for i in range(0, k):
    process_level(k-i)

print("completed: ", len(completed))
for id in completed:
    print(id)

# print("failed: ", len(failed))
# for id in failed:
#     print(id)

# print("unprocessed: ", len(comb_list))
# for id in comb_list:
#     print(id)

print("final_matrix:")
print(matrix)

validate_pattern_set(completed, len(completed))
print("Patterns Stored: ", len(completed))

test_pattern_set = [   [1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 1, 0, 0],
                       [1, 0, 1, 0, 1, 0, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0, 0, 1, 0],
                       [0, 1, 1, 0, 0, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 1, 1, 0, 0, 0, 0],
                       [1, 0, 0, 1, 0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 1, 0, 1, 0, 0],
                       [1, 0, 0, 0, 0, 0, 1, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 1, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 1, 0, 1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 1, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0, 1, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 1, 1, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 1, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1]
                    ]

# validate_pattern_set(test_pattern_set, len(test_pattern_set))

test_pattern_set = [    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],

                        # [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        # [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        # [0, 0, 1, 0, 0, 1, 0,  0, 1, 0, 0, 0],
                        # [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],

                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
                    ]
# validate_pattern_set(test_pattern_set, len(test_pattern_set))
test_pattern_set = [    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        
                        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],

                        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],

                        # [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                    ]   
# validate_pattern_set(test_pattern_set, len(test_pattern_set))


test_pattern_set = [    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],

                        # [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        # [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],

                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                        
                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],

                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],                        
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],

                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],

                        # [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        # [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        # [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                        # [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        # [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],

                        # [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
                    ]
validate_pattern_set(test_pattern_set, len(test_pattern_set))


test_pattern_set = [    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],

                        # 2-2-0
                        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],

                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],

                        # [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        # [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                        # [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],

                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        # [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],

                        # [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        # [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        # [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],

                        # [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                        # [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
                        # [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],

                        # [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        # [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                        # [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                        # [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],

                        # [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        # [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                        # [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                        # [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],

                        # [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                        # [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                        # [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                        # [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],            

                        # [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        # [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                        # [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                        # [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],

                        # [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        # [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        # [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                        # [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        # [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                        # [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],

                        # [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        # [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        # [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                        # [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                        # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                        # [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],

                        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                        # [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
                        # [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
                        # [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                        # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],

                        # [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                        # [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1], #Block 2, 1,3
                        # [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], #Block 1, 2,4
                        # [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                        # [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                        # [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                        # [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                        # [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
                        # [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],

                        # [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        # [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        # [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                        # [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                        # [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                        # [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        # [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        # [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        # [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                        # [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                        # [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                        # [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],

                        # [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                        # [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                        # [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                        # [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                        # [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                        # [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                        # [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                        # [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                        # [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        # [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                        # [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                        # [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
                    ]
# validate_pattern_set(test_pattern_set, len(test_pattern_set))