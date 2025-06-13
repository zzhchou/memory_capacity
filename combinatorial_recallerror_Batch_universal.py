import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.animation as animation
import math
import random
import itertools
import time
import json
import glob
import psutil
import copy
from collections import Counter
from collections import defaultdict

NCELL_LIST = [12]       # number of cells (neurons)
NPATT_LIST = [2,4,6,8,10,12,14,16,18,20,30,40,50,60,62, 64, 66,70,80,90,100]        # number of patterns
SPATT_LIST = [0.5]      # number of active cells per pattern


# NCELL_LIST = np.linspace(100, 500, 5, endpoint=True, dtype=int).tolist()
# NPATT_LIST = np.linspace(4, 500, 50, endpoint=True, dtype=int).tolist()
# SPATT_LIST = np.linspace(0.05, 0.2, 4, endpoint=True).tolist()

NPATT_LIST = [20]

THRESHOLD_RATIO = 0.90
ERROR_THRESHOLD = 5
CELL_ADD = 1

# random.seed(9990)
random.seed(9991)
# random.seed(9992)
# random.seed(9993)

# def prepare_animation(bar_container):
#     def animate(frame_number):
#         data = np.random.randn(1000)
#         n, _ = np.histogram(data, HIST_BINS)
#         for count, rect in zip(n, bar_container.patches):
#             rect.set_height(count)
#         return bar_container.patches
#     return animate

def generate_combinatorial_patterns(n, k):
    elements = np.arange(n)
    combinations = list(itertools.combinations(elements, k))
    comb_arr = np.array(combinations)
    
    comb_list = []
    for indexes in comb_arr:
        id = np.zeros(n, dtype=np.int32)
        for index in indexes:
            id[index] = 1
        comb_list.append(id)
    
    combinatorial_patterns = []
    for i in range(0, k):
        level_add, comb_list = pop_level(k-i, n, k, comb_list)
        combinatorial_patterns += level_add
        
#    pattern_list = [np.ndarray.tolist(pattern_array) for pattern_array in combinatorial_patterns]
    pattern_indices = []
    for pattern in combinatorial_patterns:
        pattern_indices.append([x for x in range(len(pattern)) if pattern[x] == 1])
    return pattern_indices

def pop_list(block, level, k, comb_list):
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
    return candidates, comb_list

def pop_block(block, level, k, comb_list):
    block_candidates, comb_list = pop_list(block, level, k, comb_list)
    return block_candidates, comb_list

def pop_level(level, n, k, comb_list):
    candidates = []
    for i in range(0, n//k):
        block_add, comb_list = pop_block(i, level, k, comb_list)
        candidates += block_add
    return candidates, comb_list

def generate_exhaustive_patterns(N, S):
    indices = list(range(N))
    combinations = itertools.combinations(indices, S)
    exhaustive_pattern_list = []

    # for combo in combinations:
    #     pattern = [0]*N
    #     for index in combo:
    #         pattern[index] = 1
    #     exhaustive_pattern_list.append(pattern)
    
    exhaustive_pattern_list = list(combinations)
    return exhaustive_pattern_list

def add_new_pattern(p, w, NCELL, SPATT, ACTIVATION_THRESHOLD):
    # RANDOM
    # print ('Old Pattern List: \n', p)    
    NTRIALS = 1
    NADD = 1
    RAND_LIST = [1]

    average_overlap_list = []
    max_overlap_list = []
    min_overlap_list = []
    for k in range(len(RAND_LIST)):
        overlap_list = []
        novel_pct_error_list = []
        novel_cpa_list = []
        novel_max_list = []
        novel_min_list = []
        stored_max_list = []
        stored_min_list = []
        stored_pct_error_list = []
        stored_cpa_list = []


        for j in range(NTRIALS):
            for i in range(NADD):
                new_patt = np.zeros((NCELL,NADD), dtype=bool)
                # Random Generation
                # pr = np.random.permutation(NCELL)
                # pi = pr[:SPATT]

                # Orthogonal Generation
                pattern_overlap = p.sum(axis=1)
                enumerated_list = list(enumerate(pattern_overlap))
                sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=True)         #reverse=True to maximize overlap
                sorted_indices = [item[0] for item in sorted_list]

                randomness_factor = RAND_LIST[k]/100
                shuffle_range = SPATT+int(randomness_factor*(NCELL-SPATT))

                shuffled_list = sorted_indices[:shuffle_range]
                np.random.shuffle(shuffled_list)
                sorted_indices[:shuffle_range] = shuffled_list
                pi = sorted_indices[:SPATT]

                new_patt[pi, i] = 1

                new_p = np.column_stack((p,new_patt))
                wtemp = w + np.outer(new_patt[:, i], new_patt[:, i])
                wtemp = wtemp > 0

            # print('Trial Patterns: ', np.transpose(new_patt))
            average_overlaps, novel_pct_error, novel_cpa, stored_pct_error, stored_cpa, novel_max, novel_min, stored_max, stored_min = calculate_separate_errors(ACTIVATION_THRESHOLD, w, wtemp, p, new_patt)
            overlap_list.append(average_overlaps)
            novel_max_list.append(novel_max - novel_pct_error)
            novel_min_list.append(novel_pct_error - novel_min)
            stored_max_list.append(stored_max - stored_pct_error)
            stored_min_list.append(stored_pct_error - stored_min)
            novel_pct_error_list.append(novel_pct_error)
            stored_pct_error_list.append(stored_pct_error)
            novel_cpa_list.append(novel_cpa)
            stored_cpa_list.append(stored_cpa)

        mean_overlap = sum(overlap_list)/len(overlap_list)
        average_overlap_list.append(mean_overlap)
        max_overlap_list.append(max(overlap_list) - mean_overlap)
        min_overlap_list.append(mean_overlap - min(overlap_list))

        plt.figure()
        plt.errorbar(overlap_list, novel_pct_error_list, yerr=[novel_min_list, novel_max_list], c='tab:blue', label='New Pattern', fmt='o', capsize=3)
        plt.title("Novel Pattern Overlap vs Recall Error %")
        plt.xlabel("Average Novel Pattern Overlap with Old Patterns")
        plt.ylabel("Recall Error (%)")
        plt.legend()
        plt.show()

        plt.figure()
        plt.errorbar(overlap_list, stored_pct_error_list, yerr=[stored_min_list, stored_max_list], c='tab:orange', label='Old Patterns', fmt='o', capsize=3)
        plt.title("Novel Pattern Overlap vs Recall Error %")
        plt.xlabel("Average Novel Pattern Overlap with Old Patterns")
        plt.ylabel("Recall Error (%)")
        plt.legend()
        plt.show()

    plt.figure()
    plt.errorbar(RAND_LIST, average_overlap_list, yerr = [min_overlap_list, max_overlap_list], fmt='-o', capsize=3)
    plt.title("Average Overlap of Novel Pattern based on Shuffle Percentage")
    plt.xlabel("Shuffle Percentage")
    plt.ylabel("Average Novel Pattern Overlap with Old Patterns")
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.scatter(novel_cpa_list, novel_pct_error_list, c='tab:blue', label='New Pattern')
    # plt.scatter(novel_cpa_list, stored_pct_error_list, c='tab:orange', label='Old Patterns')
    # plt.title("Cross Pattern Activation vs Recall Error %")
    # plt.xlabel("Average Cross Pattern Activation")
    # plt.ylabel("Recall Error(%)")
    # plt.xlim((15,20))
    # plt.legend()
    # plt.show()  

    # plt.figure()
    # plt.scatter(overlap_list, novel_cpa_list, c='tab:blue', label='New Pattern')
    # plt.scatter(overlap_list, stored_cpa_list, c='tab:orange', label='Old Patterns')
    # plt.title("Average Cross-Pattern Activation")
    # plt.xlabel("Average Novel Pattern Overlap")
    # plt.ylabel("Recall Error (%)")
    # plt.ylim((15,20))
    # plt.legend()
    # plt.show()


def calculate_separate_errors(ACTIVATION_THRESHOLD, w, new_w, p, new_patt):
    NPATT = np.size(p,axis=1)
    NADD = np.size(new_patt,axis=1)

    total_overlaps = []
    for i in range(NADD):
        # print(p)
        pattern_overlap = p.sum(axis=1)
        # print(pattern_overlap)
        overlap_product = np.multiply(pattern_overlap, np.transpose(new_patt[:, i]))
        total_overlaps.append(overlap_product.sum(axis=0))
    average_overlaps = sum(total_overlaps)/NADD

    old_pct_error, old_cpa, old_max, old_min = calculate_errors(NPATT, ACTIVATION_THRESHOLD, w, p)
    stored_pct_error, stored_cpa, stored_max, stored_min = calculate_errors(NPATT, ACTIVATION_THRESHOLD, new_w, p)
    novel_pct_error, novel_cpa, novel_max, novel_min = calculate_errors(NADD, ACTIVATION_THRESHOLD, new_w, new_patt)
    return average_overlaps, novel_pct_error, novel_cpa, stored_pct_error, stored_cpa, novel_max, novel_min, stored_max, stored_min

def generate_patterns(NCELL_LIST, NPATT_LIST, SPATT_LIST):
    NTRIALS = 1
    NGEN = 2
    capacity_heatmap = [[0]*len(NCELL_LIST) for _ in range(len(SPATT_LIST))]
    RAND_LIST = [0]
    THRESH_LIST = [0.75, 0.8, 0.85, 0.9, 0.95]

    THRESH_LIST = [0.7, 0.8]
    THRESHOLD_RATIO = 0.5
    RAND_LIST = [1, 2, 3, 4, 5]
    # RAND_LIST = ['B1', 'B2', 'B3', 'B4', 'B5']
    RAND_LIST = [0]


    ii = 0
    for NCELL in NCELL_LIST:
        jj = 0
        for SPATT_RATIO in SPATT_LIST:
            ERROR_LIST = []            
            ACTIVATION_THRESHOLD = int(NCELL*SPATT_RATIO*THRESHOLD_RATIO)

            if ACTIVATION_THRESHOLD == 0:
                print("WARNING: threshold set to 0")
            for NPATT in NPATT_LIST:

                plt.figure()              
                SPATT = int(NCELL*SPATT_RATIO)
                FWGT = 'wgtsN{}S{}P{}.dat'.format(NCELL, SPATT, NPATT)   # weights file
                FPATT = 'pattsN{}S{}P{}.dat'.format(NCELL, SPATT, NPATT)   # patterns file

                saveFolder = 'pyWeights/'

                np.random.seed(0)


                pattern_completion_list = []
                pattern_separation_list = []
                activation_threshold_list = []
                all_weight_matrices = []
                all_input_matrices = []
                all_overlap_traces = []
                all_mutual_info = []
                breakpoint_indices = []
                breakpoint_weight_matrix = []
                pre_breakpoint_weight_matrix = []
                post_breakpoint_output = []
                post_breakpoint_input = []
                post_breakpoint_pre_threshold = []
                pre_breakpoint_pre_threshold = []
                pre_breakpoint_output = []
                breakpoint_pattern_set = []
                breakpoint_overlap = []                
                mutual_info_15 = []
                pattern_overlap_15 = []
                initial_recall_error = []
                final_recall_error = []
                initial_cpa = []
                final_cpa = []
                cpa_15 = []
                shared_input_list = []

                exhaustive_max_cpa_9 = []
                exhaustive_min_cpa_9 = []
                exhaustive_max_overlap_9 = []
                exhaustive_cell_usage_9 = []
                exhaustive_wtemp_9 = []

                cpa_dictionaries = []

                errorbar_offset = -1
                # for THRESHOLD in THRESH_LIST:
                #     ACTIVATION_THRESHOLD = int(NCELL*SPATT_RATIO*THRESHOLD)
                #     RAND_FACTOR  = 0
                trial_num = 0
                for RAND_FACTOR in RAND_LIST:
                    trial_num = trial_num + 1
                    # NTRIALS = 1
                    # NGEN = 10
                    errorbar_offset += 1
                    trials_list = [[] for _ in range(NPATT)]



                    # num_patt_list = []
                    recall_error_list = []
                    recall_error_low = []
                    recall_error_high = []
                    error_std = []

                    min_mutual_info = [[] for _ in range(NPATT)]
                    max_mutual_info = [[] for _ in range(NPATT)]
                    mean_mutual_info = [[] for _ in range(NPATT)]
                    std_mutual_info = [[] for _ in range(NPATT)]

                    min_cpa = [[] for _ in range(NPATT)]
                    max_cpa = [[] for _ in range(NPATT)]
                    mean_cpa = [[] for _ in range(NPATT)]
                    std_cpa = [[] for _ in range(NPATT)]

                    max_CPA_counts = []
                    min_CPA_counts = []
                
                    best_candidate_counts = []
                    min_overlap = []
                    max_overlap = []
                    mean_overlap = []

                    io_mutual_info = [[] for _ in range(NPATT)]
                    normalized_io_mutual_info = [[] for _ in range(NPATT)]


                    for kk in range(NTRIALS):
                        pattern_overlap_tracker = np.zeros((NCELL, NPATT))
                        exhaustive_pattern_list = generate_exhaustive_patterns(NCELL, SPATT)
                        combinatorial_pattern_list = generate_combinatorial_patterns(NCELL, SPATT)


                        w = np.zeros((NCELL, NCELL), dtype=float)
                        w_temp = np.zeros((NCELL, NCELL), dtype=float)
                        p = np.zeros((NCELL, NPATT), dtype=bool)
                        p_out = np.zeros((NCELL, NPATT), dtype=bool)
                        pre_threshold_out = np.zeros((NCELL, NPATT), dtype=bool)
                        p_trash = np.zeros((NCELL, NPATT), dtype=bool)
                        pre_threshold_out_9 = np.zeros((NCELL, NPATT), dtype=bool)


                        total_error_count = [0]*NPATT
                        num_patt_list = []
                        plateau_flag = 0
                        breakpoint_flag = 0
                        i = 0
                        # num_best_candidates = 0           
                        while i < NPATT:
                            time1 = time.time()
                            best_pi = 0
                            best_recall_error = 1
                            best_overlap = 0

                            print ("Generating Pattern {}".format(i))
                            p[:,i] = 0
                            p_out[:,i] = 0

                            pattern_overlap = p.sum(axis = 1)
                            pattern_overlap_tracker[:, i] = pattern_overlap

                            product_sum_threshold = ((i+1)*SPATT*SPATT/NCELL)*1
                            product_sum_threshold = 100000
                            # print(product_sum_threshold)
                            vector_product_sums = [np.sum(pattern_overlap[np.asarray(vector)]) for vector in exhaustive_pattern_list]
                            filtered_sums = [product_sum for product_sum in vector_product_sums if product_sum < product_sum_threshold]
                            sorted_vectors = [vector for _, vector in sorted(zip(filtered_sums, exhaustive_pattern_list), reverse=False)]
                            sorted_sums = [sum for sum, _ in sorted(zip(filtered_sums, exhaustive_pattern_list), reverse=False)]

                            exhaustive_filtered_list = sorted_vectors


                            NGEN = len(exhaustive_filtered_list)
                            print("{} Possible Patterns".format(NGEN))


                            skip_flag = 0
                            tie_count = 0
                            num_best_candidates = 0
                            for idx in range(NGEN):
                                time2 = time.time()

                                pi = 0
                                # # new_patt = np.zeros((NCELL,1))
                                # duplicate_flag = 1

                                # # generate pattern randomly
                                # pr = np.random.permutation(NCELL)
                                # pi = pr[:SPATT]  # indices of active cells in pattern

                                # AVOID EXHAUSTIVE SEARCH
                                plateau_flag = 1

                                # # Orthogonal Generation
                            
                                if plateau_flag == 0:
                                    skip_flag = 1
                                   
                                    pattern_overlap = p.sum(axis=1)
                                    p_indices = [[index for index, value in enumerate(vector) if value == 1] for vector in p.T]
                                    p_indices = [_ for _ in p_indices if _ != []]
                                    enumerated_list = list(enumerate(pattern_overlap))
                                    sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=False)         #reverse=True to maximize overlap
                                    sorted_indices = [item[0] for item in sorted_list]

                                    if i < NCELL/SPATT:
                                        unsorted_pi = sorted_indices[:SPATT]

                                    elif i >= NCELL/SPATT:
                                        unsorted_pi = []

                                        for ii in range(SPATT):
                                            pattern_id = ii%i

                                            # pattern_overlap = p.sum(axis=1)
                                            vector_product_sums = [np.sum(pattern_overlap[np.asarray(vector)]) for vector in p_indices]
                                            sorted_vectors = [vector for _, vector in sorted(zip(vector_product_sums, p_indices), reverse=False)]

                                            sorted_p_indices = sorted(sorted_vectors[0], key=lambda x: pattern_overlap[x])
                                            unsorted_pi.append(sorted_p_indices[0])
                                            pattern_overlap[sorted_p_indices[0]] += 1
                                            # first_present_index = next((index for index in sorted_indices if (index in indices_of_ones) and (index not in unsorted_pi)), None)
                                            # if first_present_index != None:
                                            #     unsorted_pi.append(first_present_index)
                                                
                                    unsorted_set = set(unsorted_pi)

                                    for patt in exhaustive_pattern_list:
                                        exhaustive_set = set(patt)
                                        if exhaustive_set == unsorted_set:
                                            pi = patt
                                            break

                                    # pi = exhaustive_pattern_list[0]


                                # randomness_factor = RAND_FACTOR/100
                                # # shuffle_range = SPATT+int(randomness_factor*(NCELL-SPATT))
                                # shuffle_range = SPATT+int(randomness_factor*(SPATT))
                                # attempts = 0
                                # while duplicate_flag:
                                #     attempts = attempts + 1
                                #     if attempts > 10:
                                #         break
                                #     duplicate_flag = 0                                    
                                #     new_patt = np.zeros((NCELL,1))

                                #     # sorted_indices = [item[0] for item in sorted_list]

                                #     # shuffled_list = sorted_indices[SPATT:]
                                #     # np.random.shuffle(shuffled_list)
                                #     # sorted_indices[SPATT:] = shuffled_list                                

                                #     # shuffled_list = sorted_indices[:shuffle_range]
                                #     # np.random.shuffle(shuffled_list)
                                #     # sorted_indices[:shuffle_range] = shuffled_list
                                #     # if i >= NCELL/SPATT:
                                #     #     pi = []
                                #     #     for ii in range(SPATT):
                                #     #         pattern_id = ii%i
                                #     #         p_vector = np.asarray(p[:,pattern_id])
                                #     #         indices_of_ones = np.where(p_vector ==1)[0]
                                #     #         first_present_index = next((index for index in sorted_indices if (index in indices_of_ones) and (index not in pi)), None)
                                #     #         if first_present_index != None:
                                #     #             pi.append(first_present_index)
                                #     # else:
                                #     #     pi = sorted_indices[:SPATT]

                                #     # generate pattern randomly
                                #     pr = np.random.permutation(NCELL)
                                #     pi = pr[:SPATT]  # indices of active cells in pattern

                                #     new_patt[pi, 0] = 1

                                #     # print(new_patt[:,0])             

                                #     for col in p.T:
                                #         if np.array_equal(new_patt[:,0], col):
                                #             # print (new_patt[:,0], col)
                                #             duplicate_flag = 1
                                #             # if attempts > 10:
                                #             #     shuffle_range = shuffle_range + 1
                                #             #     if shuffle_range > NCELL:
                                #             #         duplicate_flag = 0
                                #             #         print("No new patterns possible: ", i)
                                #             #     attempts = 0

                                pi = 0
                                if pi == 0:
#                                    pi = combinatorial_pattern_list[idx]
                                    pi = exhaustive_filtered_list[idx]
                                    # if i > 4 and i < 15:
                                    #     random_pattern_idx = random.randrange(NGEN)
                                    #     print('random pattern idx', random_pattern_idx)
                                    #     pi = exhaustive_filtered_list[random_pattern_idx]
                                    #     skip_flag = 1
                                p[:,i] = 0
                                p[pi, i] = 1
                                # print(pi, p[:,i])
                                # store in weight matrix
                                w_temp = w+np.outer(p[:, i], p[:, i])
                                w_temp = w_temp > 0
                                for IP_ID in range(0, i):
                                    average_percent_error, average_crosspattern_activation, max_crosspattern_activation, min_crosspattern_activation, cpa_counts, max_error, min_error, input_pattern, output_pattern, pre_threshold = calculate_specific_error(IP_ID, ACTIVATION_THRESHOLD, w_temp, p)
                                    p_out[:, IP_ID] = output_pattern
                                time6 = time.time()
                                # print (p_out)
                                io_mutual = 0
                                max_duplicate = 0

                                duplicate_count_array = [0]*(i+1)
                                unique_pattern_list = []
                                unique_pattern_index = []
                                for uidx, upatt in enumerate(p_out.T):
                                    upatt = np.ndarray.tolist(upatt)
                                    if sum(upatt) == 0:
                                        break
                                    # print('unique pattern counts: ', len(upatt), len(unique_pattern_list))
                                    if upatt in unique_pattern_list:
                                        unique_pattern_index[unique_pattern_list.index(upatt)].append(uidx)
                                    else:
                                        unique_pattern_list.append(upatt)
                                        unique_pattern_index.append([uidx])
                                for uidx_list in unique_pattern_index:
                                    if uidx_list == []:
                                        duplicate_count = 0
                                    else:
                                        duplicate_count = len(uidx_list)
                                    for uidx in uidx_list:
                                        duplicate_count_array[uidx] = duplicate_count


                                for IP_ID in range(0,i):
                                    p_x = 1/(i+1)
                                    # for OP_ID in range(0, i):
                                    duplicate_count = duplicate_count_array[IP_ID]                                             
                                    p_y = duplicate_count/(i+1)
                                    p_xy = 1/(i+1)

                                    # print(p_xy, p_x, p_y, duplicate_count)
                                    io_mutual += (p_xy*math.log2(p_xy/(p_x*p_y)))/duplicate_count
                                    if duplicate_count > max_duplicate:
                                        max_duplicate = duplicate_count
                                time7 = time.time()

                                APE_list_exhaustive = []
                                CPA_list_exhaustive = []
                                pre_threshold_out_9 = np.zeros((NCELL, NPATT), dtype=int)

                                for IP_ID in range(0, i+1):
                                    average_percent_error, average_crosspattern_activation, max_crosspattern_activation, min_crosspattern_activation, cpa_counts, max_error, min_error, input_pattern, output_pattern, pre_threshold = calculate_specific_error(IP_ID, ACTIVATION_THRESHOLD, w_temp, p)
                                    pre_threshold_out_9[:, IP_ID] = pre_threshold
                                    p_out[:, IP_ID] = output_pattern
                                    APE_list_exhaustive.append(average_percent_error)
                                    CPA_list_exhaustive.append(max_crosspattern_activation)

                                # recall_error_exhaustive = np.mean(APE_list_exhaustive)
                                recall_error_exhaustive = np.mean(CPA_list_exhaustive)
                                # recall_error_exhaustive = max(CPA_list_exhaustive)
                                pattern_overlap = p.sum(axis=1)
                                pattern_overlap_max = max(pattern_overlap)

                                max_shared_inputs = []
                                min_shared_inputs = []
                                mean_shared_inputs = []

                                
                                if i == 9:
                                    shared_inputs_all_combos = []
                                    for (iidx, pattern1_9), (jjdx, pattern2_9) in itertools.combinations(enumerate(p.T), 2):
                                        shared_inputs = [ii for ii, (a, b) in enumerate(zip(pattern1_9, pattern2_9)) if a==1 and b==1]
                                        num_shared_inputs = len(shared_inputs)
                                        # if num_shared_inputs>0:
                                            # print(num_shared_inputs)
                                        shared_inputs_all_combos.append(num_shared_inputs)
                                    shared_input_list.append(np.mean(shared_inputs_all_combos))   

                                    exhaustive_max_cpa_9.append(max_crosspattern_activation)
                                    exhaustive_min_cpa_9.append(min_crosspattern_activation)
                                    exhaustive_max_overlap_9.append(max(shared_input_list))
                                    exhaustive_cell_usage_9.append(pattern_overlap_max)
                                    exhaustive_wtemp_9.append(pre_threshold_out_9)

                                    cell_usage_9 = pattern_overlap
                                    overlap_9 = shared_input_list
                                    min_overlap_9 = min(shared_input_list)
                                    mean_overlap_9 = np.mean(shared_input_list)
                                    max_overlap_9 = max(shared_input_list)
                                    min_cpa_9 = min_crosspattern_activation
                                    min_counts_9 = cpa_counts[min_crosspattern_activation]
                                    max_cpa_9 = max_crosspattern_activation
                                    max_counts_9 = cpa_counts[max_crosspattern_activation]
                                    mean_cpa_9 = average_crosspattern_activation
                                    recall_error_9 = average_percent_error

                                # print(idx)
                                if idx == 0:
                                    best_pi = pi
                                    best_w_temp = w_temp
                                    best_recall_error = recall_error_exhaustive
                                    best_overlap = pattern_overlap_max
                                elif recall_error_exhaustive < best_recall_error:
                                    # print("COMPARE: ", recall_error_exhaustive, best_recall_error)
                                    best_pi = pi
                                    best_w_temp = w_temp
                                    best_recall_error = recall_error_exhaustive
                                    best_overlap = pattern_overlap_max
                                    continue
                                elif recall_error_exhaustive == best_recall_error:
                                    # p1 = p
                                    # p2 = p
                                    # p2[:, i] = 0
                                    # p2[best_pi, i] = 1
                                    tie_flag = 1
                                    tie_count += 1
                                    decreasing_threshold = ACTIVATION_THRESHOLD
                                    while tie_flag == 1:
                                        decreasing_threshold = decreasing_threshold - 1
                                        if decreasing_threshold <= 1:
                                            break
                                        p1_APE_list = []
                                        p2_APE_list = []
                                        p1_CPA_list = []
                                        p2_CPA_list = []
                                        p1_min_count = 0
                                        p2_min_count = 0
                                        p1_max_count = 0
                                        p2_max_count = 0
                                        p1_lowest_cpa = 99
                                        p2_lowest_cpa = 99
                                        p1_highest_cpa = 0
                                        p2_highest_cpa = 0
                                        p1_Dicts = []
                                        p2_Dicts = []
                                        p1_CPA_dict = {}
                                        p2_CPA_dict = {}
                                    
                                        for IP_ID in range(0, i+1):
                                            p1 = p
                                            p1_APE, p1_CPA, p1_CPA_max, p1_CPA_min, p1_cpa_counts, p1_max_error, p1_min_error, p1_input_pattern, p1_output_pattern, pre_threshold = calculate_specific_error(IP_ID, decreasing_threshold, w_temp, p1)
                                            p2 = p
                                            p2[:, i] = 0
                                            p2[best_pi, i] = 1
                                            p2_APE, p2_CPA, p2_CPA_max, p2_CPA_min, p2_cpa_counts, p2_max_error, p2_min_error, p2_input_pattern, p2_output_pattern, pre_threshold = calculate_specific_error(IP_ID, decreasing_threshold, best_w_temp, p2)

                                            if p1_CPA_min < p1_lowest_cpa:
                                                p1_lowest_cpa = p1_CPA_min
                                            if p2_CPA_min < p2_lowest_cpa:
                                                p2_lowest_cpa = p2_CPA_min

                                            if p1_CPA_max < p1_highest_cpa:
                                                p1_highest_cpa = p1_CPA_max
                                            if p2_CPA_max < p2_highest_cpa:
                                                p2_highest_cpa = p2_CPA_max

                                            for key1, count1 in p1_cpa_counts.items():
                                                p1_CPA_dict[key1] = p1_CPA_dict.get(key1, 0) + count1
                                            for key2, count2 in p2_cpa_counts.items():
                                                p2_CPA_dict[key2] = p1_CPA_dict.get(key2, 0) + count2                                                

                                            p1_APE_list.append(p1_APE)
                                            p2_APE_list.append(p2_APE)
                                            p1_CPA_list.append(p1_CPA)
                                            p2_CPA_list.append(p2_CPA)
                                            p1_Dicts.append(p1_cpa_counts)
                                            p2_Dicts.append(p2_cpa_counts)
                                        
                                        all_keys = sorted(set(p1_CPA_dict.keys()).union(p2_CPA_dict.keys()))
                                        all_values = [(key, p1_CPA_dict.get(key, 0), p2_CPA_dict.get(key, 0)) for key in all_keys]


                                        for IP_ID in range(0, i+1):
                                            try:
                                                p1_min_count += p1_Dicts[IP_ID][p1_lowest_cpa]
                                                p1_max_count += p1_Dicts[IP_ID][p1_highest_cpa]
                                            except KeyError:
                                                print('Minimum key not found in this list')
                                            try:
                                                p2_min_count += p2_Dicts[IP_ID][p2_lowest_cpa]
                                                p2_max_count += p2_Dicts[IP_ID][p2_highest_cpa]
                                            except KeyError:
                                                print('Minimum key not found in this list')



                                        p1_APE_mean = np.mean(p1_APE_list)
                                        p2_APE_mean = np.mean(p2_APE_list)

                                        p1_CPA_mean = np.mean(p1_CPA_list)
                                        p2_CPA_mean = np.mean(p2_CPA_list)                 

                                        for key_idx in range(0, len(all_values)):
                                            value1 = all_values[key_idx][1]
                                            value2 = all_values[key_idx][2]
                                            if value1 < value2:
                                                best_pi = pi
                                                best_w_temp = w_temp
                                                best_recall_error = recall_error_exhaustive
                                                break
                                            elif value2 > value2:
                                                break
                                        
                                            num_best_candidates += 1
                                            print('Tied Pattern')

                                        # if p1_lowest_cpa != p2_lowest_cpa:
                                        #     tie_flag = 0
                                        #     if p1_lowest_cpa > p2_lowest_cpa:
                                        #         # print("COMPARE!: ", p1_lowest_cpa, p2_lowest_cpa)                                     
                                        #         best_pi = pi
                                        #         best_w_temp = w_temp
                                        #         best_recall_error = recall_error_exhaustive
                                        # elif p1_highest_cpa != p2_highest_cpa:
                                        #     tie_flag = 0
                                        #     if p1_highest_cpa < p2_highest_cpa:
                                        #         # print("COMPARE!: ", p1_highest_cpa, p2_highest_cpa)                                     
                                        #         best_pi = pi
                                        #         best_w_temp = w_temp
                                        #         best_recall_error = recall_error_exhaustive                                          
                                        # elif p1_min_count != p2_min_count:
                                        #     tie_flag = 0
                                        #     if p1_min_count < p2_min_count:
                                        #         # print("COMPARE!: ", p1_min_count, p2_min_count)                                     
                                        #         best_pi = pi
                                        #         best_w_temp = w_temp
                                        #         best_recall_error = recall_error_exhaustive
                                        # elif p1_max_count != p2_max_count:
                                        #     tie_flag = 0
                                        #     if p1_max_count < p2_max_count:
                                        #         # print("COMPARE!: ", p1_max_count, p2_max_count)                                     
                                        #         best_pi = pi
                                        #         best_w_temp = w_temp
                                        #         best_recall_error = recall_error_exhaustive 
                                        # if p1_CPA_mean != p2_CPA_mean:
                                        #     tie_flag = 0
                                        #     if p1_CPA_mean < p2_CPA_mean:                                      
                                        #         best_pi = pi
                                        #         best_w_temp = w_temp
                                        #         best_recall_error = recall_error_exhaustive                       
                                        # if p1_APE_mean != p2_APE_mean:
                                        #     tie_flag = 0
                                        #     if p1_APE_mean < p2_APE_mean:                                      
                                        #         best_pi = pi
                                        #         best_w_temp = w_temp
                                        #         best_recall_error = recall_error_exhaustive
                                else:
                                    pi = best_pi
                                    w_temp = best_w_temp
                                if skip_flag == 1:
                                    skip_flag = 0
                                    break

                            
                            time3 = time.time()
                            # print('Done testing all patterns: {}'.format(time3-time1))
                            # print('1-Cycle Output Patterns calculated: {}'.format(time6-time2))

                            # print('1-Cycle Mutual Info Calculated: {}'.format(time7-time6))                            # w -= np.outer(p[:, i], p[:, i])
                            # print(best_pi)
                            # print("Tried {} Patterns".format(idx+1))
                            # print("Ties: ", tie_count)

                            p[:,i] = 0
                            p[best_pi, i] = 1

                            p_trash[:,i] = 1

                            exhaustive_pattern_list.remove(best_pi)

                            pattern_overlap = p.sum(axis=1)/(i+1)
                            min_overlap.append(min(pattern_overlap))
                            max_overlap.append(max(pattern_overlap))
                            mean_overlap.append(np.mean(pattern_overlap))

                            best_candidate_counts.append(num_best_candidates)

                            old_w = w > 0

                            w += np.outer(p[:, i], p[:, i])
                            # w += np.outer(p_trash[:, i], p_trash[:, i])
                            w_temp = w > 0   




                            # new_p = np.column_stack((p,new_patt))
                            # wtemp = w + np.outer(new_patt[:, 0], new_patt[:, 0])
                            # wtemp = wtemp > 0

                            p_out = np.zeros((NCELL, NPATT), dtype=bool)
                            pre_threshold_out = np.zeros((NCELL, NPATT), dtype=int)

                            PATT_ID = 0
                            total_error_count[i] = 0
                            if i >= PATT_ID:
                                # num_patt_list.append(i+1)
                                average_percent_error, average_crosspattern_activation, max_crosspattern_activation, min_crosspattern_activation, cpa_counts, max_error, min_error, input_pattern, output_pattern, pre_threshold = calculate_specific_error(PATT_ID, ACTIVATION_THRESHOLD, w_temp, p)
                                APE_list = [average_percent_error]
                                CPA_list = [average_crosspattern_activation]
                                max_CPA_list = [max_crosspattern_activation]
                                min_CPA_list = [min_crosspattern_activation]
                                num_patt_list.append(i+1)

                            
                                ones_count = [0]*NCELL
                                zeros_count = [0]*NCELL
                                mutual_info = [0]*NCELL
                                for idx in range(NCELL):
                                    for pdx in range(i):
                                        pattern1 = p[:,pdx]
                                        if pattern1[idx] == 1:
                                            ones_count[idx] += 1
                                        if pattern1[idx] == 0:
                                            zeros_count[idx] += 1
                                for idx in range(NCELL):
                                    for pdx in range(i):
                                        pattern1 = p[:,pdx]
                                        p_y = 1/(i+1)
                                        if pattern1[idx] == 1:
                                            p_x = ones_count[idx]/(i+1)
                                            p_xy = 1/ones_count[idx]
                                        elif pattern1[idx] == 0:
                                            p_x = zeros_count[idx]/(i+1)
                                            p_xy = 1/zeros_count[idx]
                                        mutual_info[idx] += p_xy*math.log2(p_xy/(p_x*p_y))
                            
                                min_mutual_info[i].append(min(mutual_info))
                                max_mutual_info[i].append(max(mutual_info))
                                mean_mutual_info[i].append(np.mean(mutual_info))
                                std_mutual_info[i].append(np.std(mutual_info))
                               
                               



                                cpa_min_counts = 0
                                cpa_max_counts = 0
                                CPA_dict = {}

                                for IP_ID in range(0, i+1):
                                    average_percent_error, average_crosspattern_activation, max_crosspattern_activation, min_crosspattern_activation, cpa_counts, max_error, min_error, input_pattern, output_pattern, pre_threshold = calculate_specific_error(IP_ID, ACTIVATION_THRESHOLD, w_temp, p)

                                    pre_threshold_out[:, IP_ID] = pre_threshold
                                    p_out[:, IP_ID] = output_pattern
                                    APE_list.append(average_percent_error)
                                    CPA_list.append(average_crosspattern_activation)
                                    max_CPA_list.append(max_crosspattern_activation)
                                    min_CPA_list.append(min_crosspattern_activation)
                                    total_error_count[i] += max_error

                                    cpa_min_counts += cpa_counts[min_crosspattern_activation]
                                    cpa_max_counts += cpa_counts[max_crosspattern_activation]


                                    for key1, count1 in cpa_counts.items():
                                        CPA_dict[key1] = CPA_dict.get(key1, 0) + count1
  
                                min_CPA_counts.append(cpa_min_counts)
                                max_CPA_counts.append(cpa_max_counts)

                                max_key = SPATT
                                for key_idx in range(0, max_key + 1):
                                    CPA_dict.setdefault(key_idx, 0)

                                # print(CPA_dict)
                                cpa_dictionaries.append(CPA_dict)

                                if i == 14:
                                    mutual_info_15.append(np.mean(mutual_info))
                                if i <=50:
                                    cpa_15.append(pre_threshold_out)

                                min_cpa[i].append(min(min_CPA_list))
                                max_cpa[i].append(max(max_CPA_list))
                                mean_cpa[i].append(np.mean(CPA_list))
                                std_cpa[i].append(np.std(CPA_list))

                                if i > 1:
                                    # print ('Breakpoint found', np.mean(APE_list), trials_list[i-1][-1], trials_list[i-2][-1])
                                    # if np.mean(APE_list) > trials_list[i-1][-1] and i>20:
                                    if np.mean(APE_list) > 0 and i>20:                                
                                        if breakpoint_flag == 0:
                                            # del breakpoint_weight_matrix[-1]
                                            # del breakpoint_pattern_set[-1]
                                            # del breakpoint_overlap[-1]

                                            breakpoint_indices.append(i)
                                            breakpoint_weight_matrix.append(w_temp)
                                            post_breakpoint_output.append(p_out)
                                            post_breakpoint_input.append(p)
                                            post_breakpoint_pre_threshold.append(pre_threshold_out)
                                            pre_breakpoint_pre_threshold.append(old_pre_threshold_out)
                                            pre_breakpoint_output.append(p_out_old)
                                            pre_breakpoint_weight_matrix.append(old_w)
                                            breakpoint_pattern_set.append(i)
                                            breakpoint_overlap.append(i)
                                            breakpoint_flag = 1
                                p_out_old = copy.deepcopy(p_out)
                                old_pre_threshold_out = copy.deepcopy(pre_threshold_out)



                                # print('Mean APE: ', np.mean(APE_list))
                                trials_list[i].append(np.mean(APE_list))
                                if i == NPATT-1:
                                    final_recall_error.append(np.mean(APE_list))
                                    final_cpa.append(np.mean(CPA_list))
                                if i == 14:
                                    initial_recall_error.append(np.mean(APE_list))
                                    initial_cpa.append(np.mean(CPA_list))
                                    # print('Avg pct error:', average_percent_error)

                                duplicate_count_array = [0]*(i+1)
                                duplicate_instance = [0]
                                unique_pattern_list = []
                                unique_pattern_index = []
                                for uidx, upatt in enumerate(p_out.T):
                                    upatt = np.ndarray.tolist(upatt)
                                    if sum(upatt) == 0:
                                        break
                                    # print('unique pattern counts: ', len(upatt), len(unique_pattern_list))
                                    if upatt in unique_pattern_list:
                                        unique_pattern_index[unique_pattern_list.index(upatt)].append(uidx)
                                    else:
                                        unique_pattern_list.append(upatt)
                                        unique_pattern_index.append([uidx])
                                for uidx_list in unique_pattern_index:
                                    if uidx_list == []:
                                        duplicate_count = 0
                                    else:
                                        duplicate_count = len(uidx_list)

                                        duplicate_instance = uidx_list
                                        if duplicate_count > 1:
                                            converging_inputs = []
                                            for uidx in uidx_list:
                                                converging_inputs.append(p.T[uidx])
                                            # print(p_out.T[uidx])
                                            # print(pattern_overlap)
                                            patt1_unique = np.logical_xor(p_out.T[uidx],p.T[uidx])
                                    for uidx in uidx_list:
                                        duplicate_count_array[uidx] = duplicate_count


                                io_mutual = 0
                                max_duplicate = 0

                                if i == 0:
                                    io_mutual = 0
                                else:
                                    for IP_ID in range(0,i+1):
                                        p_x = 1/(i+1)
                                        # for OP_ID in range(0, i):
                                        duplicate_count = duplicate_count_array[IP_ID]                                             
                                        p_y = duplicate_count/(i+1)
                                        p_xy = 1/(i+1)
                                        # print(p_xy, p_x, p_y, duplicate_count)
                                        # print(p_xy*math.log2(p_xy/(p_x*p_y))/duplicate_count)
                                        io_mutual += (p_xy*math.log2(p_xy/(p_x*p_y)))/duplicate_count
                                        if duplicate_count > max_duplicate:
                                            max_duplicate = duplicate_count


                                # for IP_ID in range(0,i):
                                #     p_x = 1/(i+1)
                                #     for OP_ID in range(0, i):
                                #         duplicate_count = 0
                                #         for col in p_out.T:
                                #             if sum(col) == 0:
                                #                 break
                                #             # print(p_out[:, OP_ID])
                                #             # print(col)
                                #             # print(np.array_equal(p_out[:, OP_ID], col))
                                #             if np.array_equal(p_out[:, OP_ID], col):
                                #                 duplicate_count += 1
                                #                 # print("Duplicate Output Found")
                                #         if duplicate_count == 0:
                                #             duplicate_count = 1
                                #         p_y = duplicate_count/(i+1)
                                #         p_xy = 1/(i+1)
                                #         io_mutual += p_xy*math.log2(p_xy/(p_x*p_y))/duplicate_count
                                #         if duplicate_count > max_duplicate:
                                #             max_duplicate = duplicate_count

                                io_mutual_info[i].append(io_mutual)
                                normalized_io_mutual_info[i].append(io_mutual/(i+1))
                                # print('{} Max Duplicated Outputs'.format(max_duplicate))
                            # print(len(io_mutual_info))    
                            # print(np.sum(io_mutual_info[i]), np.sum(io_mutual_info[i-1]))
                             
                            if len(io_mutual_info[i]) == 0:
                                plateau_flag = 0
                            elif i == 0:
                                plateau_flag = 0
                            elif np.sum(io_mutual_info[i]) > np.sum(io_mutual_info[i-1]):
                                plateau_flag = 0
                            elif plateau_flag == 0:
                                plateau_flag = 1
                                skip_flag = 0
                                del trials_list[i][-1]
                                del num_patt_list[-1]
                                del min_overlap[-1]
                                del max_overlap[-1]
                                del mean_overlap[-1]

                                del min_mutual_info[i][-1]
                                del max_mutual_info[i][-1]
                                del mean_mutual_info[i][-1]
                                del std_mutual_info[i][-1]

                                del min_cpa[i][-1]
                                del max_cpa[i][-1]
                                del mean_cpa[i][-1]
                                del std_cpa[i][-1]

                                # del min_CPA_counts[i][-1]
                                # del max_CPA_counts[i][-1]

                                del io_mutual_info[i][-1]
                                del normalized_io_mutual_info[i][-1]

                                i -= 1

                            else:
                                plateau_flag = 1
                                skip_flag = 0

                            i += 1



                        # print(new_patt[:,0])

                        # w += np.outer(p[:, i], p[:, i])
                        # if i>=PATT_ID:
                        #     mean_across_trials = sum(trials_list)/len(trials_list)
                        #     std_across_trials = np.std(trials_list)
                        #     max_across_trials = max(trials_list)
                        #     min_across_trials = min(trials_list)
                        #     num_patt_list.append(i+1)
                        #     recall_error_list.append(mean_across_trials)
                        #     recall_error_low.append(mean_across_trials - min_across_trials)
                        #     recall_error_high.append(max_across_trials - mean_across_trials)
                        #     error_std.append(std_across_trials)

                    all_weight_matrices.append(w_temp)
                    all_input_matrices.append(p)
                    all_overlap_traces.append(pattern_overlap_tracker)
                    all_mutual_info.append(mutual_info)

                    mem = psutil.virtual_memory()
                    mem_usage = mem.total - mem.available
                    mem_usage_mb = mem_usage / (1024*1024)
                    print('Memory Usage: {} MB, {}%'.format(mem_usage_mb, mem.percent))

                    p_diff = np.logical_xor(p_out,p)
                    diff_count = np.sum(p_diff, axis=1)
                    diff_avg = np.mean(diff_count)
                    unique_out = np.unique(np.transpose(p_out), axis=0)

                    pattern_completion = 100 - (diff_avg/NCELL)*100
                    pattern_separation = (len(unique_out)/NPATT)*100

                    pattern_completion_list.append(pattern_completion)
                    pattern_separation_list.append(pattern_separation)
                    activation_threshold_list.append(RAND_FACTOR)

                    recall_error_list = [sum(trials_list[i])/len(trials_list[i]) for i in range(len(trials_list))]
                    error_std = [np.std(trials_list[i]) for i in range(len(trials_list))]

                    mutual_info_list = [np.sum(mean_mutual_info[i])/len(mean_mutual_info[i]) for i in range(len(mean_mutual_info))]
                    max_mutual_info_list = [np.sum(max_mutual_info[i])/len(max_mutual_info[i]) for i in range(len(max_mutual_info))]
                    min_mutual_info_list = [np.sum(min_mutual_info[i])/len(min_mutual_info[i]) for i in range(len(min_mutual_info))]

                    cpa_list = [np.sum(mean_cpa[i])/len(mean_cpa[i]) for i in range(len(mean_cpa))]
                    max_cpa_list = [np.sum(max_cpa[i])/len(max_cpa[i]) for i in range(len(max_cpa))]
                    min_cpa_list = [np.sum(min_cpa[i])/len(min_cpa[i]) for i in range(len(min_cpa))]


                    io_mutual_info_list = [np.sum(io_mutual_info[i])/len(io_mutual_info[i]) for i in range(len(io_mutual_info))]
                    norm_io_mutual_info_list = [np.sum(normalized_io_mutual_info[i])/len(normalized_io_mutual_info[i]) for i in range(len(normalized_io_mutual_info))]

                    asymmetric_error = [recall_error_low, recall_error_high]
                    print("Pattern Completion = {}".format(pattern_completion))
                    print("Pattern Separation = {}".format(pattern_separation))
                    plt.errorbar(num_patt_list, recall_error_list, yerr=error_std, errorevery=(0+errorbar_offset,len(RAND_LIST)), capsize=3, capthick=1, label=RAND_FACTOR)
                    plt.fill_between(num_patt_list, y1=[a_i - b_i for a_i, b_i in zip(recall_error_list, error_std)], y2=[sum(x) for x in zip(recall_error_list, error_std)], alpha=0.1)




                plt.title("Recall Error Curve for Randomized Patterns (5-15)")
                plt.legend(loc='upper left')
                plt.xlabel("Number of Patterns Stored")
                plt.ylabel("Average Total Error (%)")
                plt.show()
                
                for kdx in range(len(all_weight_matrices)):
                    json_data = {
                        "num_patterns": num_patt_list,
                        "pattern_list": all_input_matrices[kdx].tolist(),
                        "weight_matrices": all_weight_matrices[kdx].tolist(),
                        "overlap_traces": all_overlap_traces[kdx].tolist(),
                        "pattern_set_mutual_info": all_mutual_info[kdx]
                    }

                    with open("saved_data_{}_{}_{}_{}.json".format(NCELL, SPATT, NPATT, RAND_LIST[kdx]), "w") as outfile:
                        json.dump(json_data, outfile)

                for jdx in range(len(all_weight_matrices)):
                    w_jdx = all_weight_matrices[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(w_jdx, aspect='auto', interpolation='none')
                    plt.xlabel('Output Cell ID #')
                    plt.ylabel('Input Cell ID #')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()

                max_mutual_info_error = [max_mutual_info_list[i] - mutual_info_list[i] for i in range(len(mutual_info_list))]
                min_mutual_info_error = [mutual_info_list[i] - min_mutual_info_list[i] for i in range(len(mutual_info_list))]
                plt.figure()
                plt.errorbar(num_patt_list, mutual_info_list, yerr=[min_mutual_info_error, max_mutual_info_error], errorevery=(0,10), capsize=3, capthick=1)
                plt.fill_between(num_patt_list, y1=min_mutual_info_list, y2=max_mutual_info_list, alpha=0.1)
                plt.title("Mutual Information between Input Cells and Input Patterns")
                plt.xlabel("Number of Patterns Stored")
                plt.ylabel("Mutual Information")
                plt.show()

                max_cpa_error = [max_cpa_list[i] - cpa_list[i] for i in range(len(cpa_list))]
                min_cpa_error = [cpa_list[i] - min_cpa_list[i] for i in range(len(cpa_list))]
                plt.figure()
                plt.errorbar(num_patt_list, cpa_list, yerr=[min_cpa_error, max_cpa_error], errorevery=(0,10), capsize=3, capthick=1)
                plt.fill_between(num_patt_list, y1=min_cpa_list, y2=max_cpa_list, alpha=0.1)
                plt.title("Cross-Pattern Activation vs # Patterns Stored")
                plt.xlabel("Number of Patterns Stored")
                plt.ylabel("Cross-Pattern Activation")
                plt.show()                

                plt.rcParams['figure.figsize'] = (9, 7)

                fig, axs = plt.subplots(5, sharex=True)
                axs[0].plot(num_patt_list, recall_error_list)
                fig.suptitle('N = {}; S = {}'.format(NCELL, SPATT))
                fig.supxlabel('Number of Patterns Stored')
                axs[0].set_ylabel('Recall Error')
                axs[1].errorbar(num_patt_list, mutual_info_list, yerr=[min_mutual_info_error, max_mutual_info_error], errorevery=(0,10), capsize=3, capthick=1)
                axs[1].fill_between(num_patt_list, y1=min_mutual_info_list, y2=max_mutual_info_list, alpha=0.1)
                # axs[1].plot(num_patt_list, mutual_info_list)
                axs[1].set_ylabel('MI')
                axs[2].errorbar(num_patt_list, cpa_list, yerr=[min_cpa_error, max_cpa_error], errorevery=(0,10), capsize=3, capthick=1)
                axs[2].fill_between(num_patt_list, y1=min_cpa_list, y2=max_cpa_list, alpha=0.1)
                # axs[2].plot(num_patt_list, cpa_list)
                axs[2].set_ylabel('CPA')
                axs[3].plot(num_patt_list, min_CPA_counts)
                axs[3].set_ylabel('# Min CPA')
                axs[4].plot(num_patt_list, max_CPA_counts)
                axs[4].set_ylabel('# Max CPA')

                plt.show()

                accumulated = defaultdict(list)
                for d in cpa_dictionaries:
                    for key, value in d.items():
                        accumulated[key].append(value)
                sorted_keys = sorted(accumulated.keys())
                plot_data = [accumulated[key] for key in sorted_keys]

                # print(plot_data)
                plt.figure()
                for key_idx in range(0, len(sorted_keys)):
                    plt.plot(num_patt_list, plot_data[key_idx])
                plt.legend(sorted_keys)
                plt.xlabel('Patterns Stored')
                plt.ylabel('CPA Counts')
                plt.show()

                print(best_candidate_counts)
                plt.figure()
                plt.plot(num_patt_list, best_candidate_counts)
                plt.xlabel('Patterns Stored')
                plt.ylabel('# Tied Patterns')
                plt.show()

                delta_mutual_info = [mutual_info_list[i+1] - mutual_info_list[i] for i in range(len(mutual_info_list)-1)]
                delta_cpa = [cpa_list[i+1] - cpa_list[i] for i in range(len(cpa_list)-1)]
                delta_recall_error = [recall_error_list[i+1] - recall_error_list[i] for i in range(len(recall_error_list)-1)]


                fig, axs = plt.subplots(3, sharex=True)
                axs[0].plot(num_patt_list[:-1], delta_recall_error)
                fig.suptitle('N = {}; S = {}'.format(NCELL, SPATT))
                fig.supxlabel('Number of Patterns Stored')
                axs[0].set_ylabel('Delta Recall Error')
                axs[1].plot(num_patt_list[:-1], delta_mutual_info)
                axs[1].set_ylabel('Delta Mutual Info')
                axs[2].plot(num_patt_list[:-1], delta_cpa)
                axs[2].set_ylabel('Delta Cross-Pattern Activation')
                plt.show()

                delta_mutual_info_norm = [delta_mutual_info[i]/(i+1) for i in range(len(delta_mutual_info))]
                delta_recall_error_norm = [delta_recall_error[i]/(i+1) for i in range(len(delta_recall_error))]
                delta_cpa_norm = [delta_cpa[i]/(i+1) for i in range(len(delta_cpa))]

                fig, axs = plt.subplots(3, sharex=True)
                axs[0].plot(num_patt_list[:-1], delta_recall_error_norm)
                fig.suptitle('N = {}; S = {}'.format(NCELL, SPATT))
                fig.supxlabel('Number of Patterns Stored')
                axs[0].set_ylabel('Normalized Delta Recall Error')
                axs[1].plot(num_patt_list[:-1], delta_mutual_info_norm)
                axs[1].set_ylabel('Normalized Delta Mutual Info')
                axs[2].plot(num_patt_list[:-1], delta_cpa_norm)
                axs[2].set_ylabel('Normalized Cross-Pattern Activation')
                plt.show()

                plt.rcParams['figure.figsize'] = (6.4,4.8)

                plt.figure()
                plt.plot(num_patt_list, io_mutual_info_list, color='orange')
                plt.title("Mutual Information between Input and Output Patterns")
                plt.xlabel("Number of Patterns Stored")
                plt.ylabel("Mutual Information")
                plt.show()

                plt.figure()
                plt.plot(num_patt_list, norm_io_mutual_info_list, color='orange')
                plt.title("Mutual Information per Pattern between Input and Output Patterns")
                plt.xlabel("Number of Patterns Stored")
                plt.ylabel("Normalized Mutual Information")
                plt.show()


                max_overlap_error = [max_overlap[i] - mean_overlap[i] for i in range(len(mean_overlap))]
                min_overlap_error = [mean_overlap[i] - min_overlap[i] for i in range(len(mean_overlap))]

                print(len(num_patt_list), len(mean_overlap))

                plt.figure()
                plt.errorbar(num_patt_list, mean_overlap, yerr=[min_overlap_error, max_overlap_error], errorevery=(0,10), capsize=3, capthick=1)
                plt.fill_between(num_patt_list, y1=min_overlap, y2=max_overlap, alpha=0.1)
                plt.title("Pattern Cell Overlap as More Patterns Stored")
                plt.xlabel("Number of Patterns Stored")
                plt.ylabel("Normalized Pattern Set Overlap")
                plt.xlim(left=5)
                plt.show()

                plt.figure()
                plt.plot(activation_threshold_list, pattern_completion_list, label='Recall Accuracy')
                plt.plot(activation_threshold_list, pattern_separation_list, label='Pattern Separation')
                plt.xlabel("Word Exchange (%)")
                plt.ylabel("Pattern Separation/Recall Accuracy")
                plt.legend()
                plt.show()

                w = w > 0  # clip weight matrix

                # print(p)

                for jdx in range(len(all_input_matrices)):
                    p_jdx = all_input_matrices[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(p_jdx, aspect='auto', interpolation='none')
                    plt.ylabel('Input Cell ID #')
                    plt.xlabel('Input Pattern #')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(p_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(p_jdx.shape[0]+1)-.5, minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    ax.set_xlim([4.5,14.5])
                    plt.show()
                
                plt.rcParams['figure.figsize'] = (18, 4.8)

                for jdx in range(len(all_input_matrices)):
                    p_jdx = all_input_matrices[jdx]
                    ax = plt.figure().gca()

                    plt.imshow(p_jdx, aspect='auto', interpolation='none')
                    plt.ylabel('Input Cell ID #')
                    plt.xlabel('Input Pattern #')
                    # plt.rcParams['figure.figsize'] = (18, 4.8)
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(p_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(p_jdx.shape[0]+1)-.5, minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()

                for jdx in range(len(post_breakpoint_input)):
                    w_jdx = post_breakpoint_input[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(w_jdx, aspect='auto', interpolation='none')
                    plt.xlabel('Input Pattern #')
                    plt.ylabel('Input Cell ID #')
                    plt.title('Post-Breakpoint Input Patterns')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    plt.xlim(-0.5, breakpoint_indices[jdx] + 0.5)

                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()

                for jdx in range(len(pre_breakpoint_output)):
                    w_jdx = pre_breakpoint_output[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(w_jdx, aspect='auto', interpolation='none')
                    plt.xlabel('Output Pattern #')
                    plt.ylabel('Input Cell ID #')
                    plt.title('Pre-Breakpoint Output Patterns')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    plt.xlim(-0.5, breakpoint_indices[jdx] + 0.5)

                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()


                for jdx in range(len(post_breakpoint_output)):
                    w_jdx = post_breakpoint_output[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(w_jdx, aspect='auto', interpolation='none')
                    plt.xlabel('Output Pattern #')
                    plt.ylabel('Input Cell ID #')
                    plt.title('Post-Breakpoint Output Patterns')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    plt.xlim(-0.5, breakpoint_indices[jdx] + 0.5)

                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()

                exhaustive_cell_usage_9_index = exhaustive_cell_usage_9.index(min(exhaustive_cell_usage_9))
                exhaustive_max_cpa_9_index = exhaustive_max_cpa_9.index(min(exhaustive_max_cpa_9))
                exhaustive_min_cpa_9_index = exhaustive_min_cpa_9.index(max(exhaustive_min_cpa_9))
                exhaustive_max_overlap_9_index = exhaustive_max_overlap_9.index(min(exhaustive_max_overlap_9))

                print(exhaustive_cell_usage_9_index, exhaustive_max_cpa_9_index, exhaustive_min_cpa_9_index, exhaustive_max_overlap_9_index)
                print(len(exhaustive_wtemp_9), len(exhaustive_wtemp_9[exhaustive_cell_usage_9_index]))
                w_jdx1 = exhaustive_wtemp_9[exhaustive_cell_usage_9_index]
                ax = plt.figure().gca()
                im = plt.imshow(w_jdx1, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                plt.xlabel('Input Pattern #')
                plt.ylabel('Output Cell ID #')
                plt.title('CPA for Min Cell Usage Patterns')
                ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                ax.set_xticks(np.arange(w_jdx1.shape[1]+1)-.5, minor=True)
                ax.set_yticks(np.arange(w_jdx1.shape[0]+1)-.5, minor=True)
                plt.xlim(-0.5, 9.5)

                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)
                fig.colorbar(im, ax=ax)
                plt.show()

                w_jdx2 = exhaustive_wtemp_9[exhaustive_max_cpa_9_index]
                ax = plt.figure().gca()
                im = plt.imshow(w_jdx2, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                plt.xlabel('Input Pattern #')
                plt.ylabel('Output Cell ID #')
                plt.title('CPA for Max CPA Pattern')
                ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                ax.set_xticks(np.arange(w_jdx2.shape[1]+1)-.5, minor=True)
                ax.set_yticks(np.arange(w_jdx2.shape[0]+1)-.5, minor=True)
                plt.xlim(-0.5, 9.5)

                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)
                fig.colorbar(im, ax=ax)
                plt.show()

                w_jdx3 = exhaustive_wtemp_9[exhaustive_min_cpa_9_index]
                ax = plt.figure().gca()
                im = plt.imshow(w_jdx3, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                plt.xlabel('Input Pattern #')
                plt.ylabel('Output Cell ID #')
                plt.title('CPA for Min CPA Patterns')
                ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                ax.set_xticks(np.arange(w_jdx3.shape[1]+1)-.5, minor=True)
                ax.set_yticks(np.arange(w_jdx3.shape[0]+1)-.5, minor=True)
                plt.xlim(-0.5, 9.5)

                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)
                fig.colorbar(im, ax=ax)
                plt.show()

                w_jdx4 = exhaustive_wtemp_9[exhaustive_max_overlap_9_index]
                ax = plt.figure().gca()
                im = plt.imshow(w_jdx4, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                plt.xlabel('Input Pattern #')
                plt.ylabel('Output Cell ID #')
                plt.title('CPA for Max Overlap Patterns')
                ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                ax.set_xticks(np.arange(w_jdx4.shape[1]+1)-.5, minor=True)
                ax.set_yticks(np.arange(w_jdx4.shape[0]+1)-.5, minor=True)
                plt.xlim(-0.5, 9.5)

                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)
                fig.colorbar(im, ax=ax)
                plt.show()

                plt.figure()
                plt.hist(exhaustive_max_cpa_9)
                plt.title('Max CPA')
                plt.show()
                plt.figure()
                plt.hist(exhaustive_min_cpa_9)
                plt.title('Min CPA')
                plt.show()
                plt.figure()
                plt.hist(exhaustive_cell_usage_9)
                plt.title('Cell Usage')
                plt.show()
                plt.figure()
                plt.hist(shared_input_list)
                plt.title('Pattern Overlap')
                plt.show()

                # for jdx in range(len(overlap_9)):
                #     w_jdx = overlap_9[jdx]
                #     ax = plt.figure().gca()
                #     im = plt.imshow(w_jdx, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                #     plt.xlabel('Input Pattern #')
                #     plt.ylabel('Output Cell ID #')
                #     plt.title('CPA for < 20 Patterns')
                #     ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                #     ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                #     ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                #     plt.xlim(-0.5, 9.5)

                #     ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                #     ax.tick_params(which='minor', bottom=False, left=False)
                #     fig.colorbar(im, ax=ax)
                #     plt.show()

                for jdx in range(len(cpa_15)):

                    w_jdx = cpa_15[jdx]
                    ax = plt.figure().gca()
                    im = plt.imshow(w_jdx, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                    plt.xlabel('Input Pattern #')
                    plt.ylabel('Output Cell ID #')
                    plt.title('CPA for < 10 Patterns')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    plt.xlim(-0.5, 9.5)

                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    fig.colorbar(im, ax=ax)
                    plt.show()

                for jdx in range(len(pre_breakpoint_pre_threshold)):

                    w_jdx = pre_breakpoint_pre_threshold[jdx]
                    ax = plt.figure().gca()
                    im = plt.imshow(w_jdx, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                    plt.xlabel('Input Pattern #')
                    plt.ylabel('Output Cell ID #')
                    plt.title('Pre-Breakpoint Pre-Threshold Activation')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    plt.xlim(-0.5, breakpoint_indices[jdx] + 0.5)

                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    fig.colorbar(im, ax=ax)
                    plt.show()

                for jdx in range(len(post_breakpoint_pre_threshold)):
                    w_jdx = post_breakpoint_pre_threshold[jdx]
                    ax = plt.figure().gca()
                    im = plt.imshow(w_jdx, aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=SPATT)
                    plt.xlabel('Input Pattern #')
                    plt.ylabel('Output Cell ID #')
                    plt.title('Post-Breakpoint Pre-Threshold Activation')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    plt.xlim(-0.5, breakpoint_indices[jdx] + 0.5)

                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    fig.colorbar(im, ax=ax)
                    plt.show()

                ax = plt.figure().gca()
                plt.imshow(p_out, aspect='auto', interpolation='none')
                plt.ylabel('Output Cell ID #')
                plt.xlabel('Output Pattern #')
                plt.rcParams['figure.figsize'] = (6.4,4.8)
                ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                ax.set_xticks(np.arange(p_out.shape[1]+1)-.5, minor=True)
                ax.set_yticks(np.arange(p_out.shape[0]+1)-.5, minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)
                plt.show()

                ax = plt.figure().gca()
                plt.imshow(abs(p_diff), aspect='auto', interpolation='none')
                plt.ylabel('Input Cell ID #')
                plt.xlabel('Cue Pattern #')
                ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                ax.set_xticks(np.arange(abs(p_diff).shape[1]+1)-.5, minor=True)
                ax.set_yticks(np.arange(abs(p_diff).shape[0]+1)-.5, minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                ax.tick_params(which='minor', bottom=False, left=False)
                plt.show()

                # ax = plt.figure().gca()
                # plt.imshow(np.transpose(p), aspect='auto', interpolation='none')
                # plt.xlabel('Cell ID #')
                # plt.ylabel('Input Pattern #')
                # ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                # ax.set_xticks(np.arange(np.transpose(p).shape[1]+1)-.5, minor=True)
                # ax.set_yticks(np.arange(np.transpose(p).shape[0]+1)-.5, minor=True)
                # ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
                # ax.tick_params(which='minor', bottom=False, left=False)
                # plt.show()

                # ax = plt.figure().gca()
                # plt.imshow(np.transpose(p_out), aspect='auto', interpolation='none')
                # plt.xlabel('Cell ID #')
                # plt.ylabel('Output Pattern #')
                # ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                # ax.set_xticks(np.arange(np.transpose(p_out).shape[1]+1)-.5, minor=True)
                # ax.set_yticks(np.arange(np.transpose(p_out).shape[0]+1)-.5, minor=True)
                # ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
                # ax.tick_params(which='minor', bottom=False, left=False)
                # plt.show()

                # ax = plt.figure().gca()
                # plt.imshow(np.transpose(abs(p_diff)), aspect='auto', interpolation='none')
                # plt.xlabel('Cell ID #')
                # plt.ylabel('Cue Pattern #')
                # ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                # ax.set_xticks(np.arange(np.transpose(abs(p_diff)).shape[1]+1)-.5, minor=True)
                # ax.set_yticks(np.arange(np.transpose(abs(p_diff)).shape[0]+1)-.5, minor=True)
                # ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
                # ax.tick_params(which='minor', bottom=False, left=False)
                # plt.show()

                overlap_tracker_T = [list(x) for x in zip(pattern_overlap_tracker)]

                for jdx in range(len(all_overlap_traces)):
                    overlap_trial = all_overlap_traces[jdx]
                    plt.figure()
                    # print('length check', NPATT, overlap_tracker_T[0])
                    for a in range(NCELL):
                        overlap_trace = overlap_trial[a]
                        plt.plot(num_patt_list, overlap_trace)
                    plt.xlabel('Number of Patterns')
                    plt.ylabel('Number of counts')
                    plt.xlim(4.5, 14.5)
                    plt.show()

                for jdx in range(len(all_overlap_traces)):
                    overlap_trial = all_overlap_traces[jdx]
                    plt.figure()
                    # print('length check', NPATT, overlap_tracker_T[0])
                    for a in range(NCELL):
                        overlap_trace = overlap_trial[a]
                        plt.plot(num_patt_list, overlap_trace)
                    plt.xlabel('Number of Patterns')
                    plt.ylabel('Number of counts')
                    plt.show()

                error_count_diff = [0]
                for idx in range(0, NPATT-1): 
                    error_count_diff.append(total_error_count[idx+1] - total_error_count[idx])
                plt.figure()
                plt.plot(num_patt_list, error_count_diff)
                plt.xlabel('Number of Patterns')
                plt.ylabel('Errors added')
                plt.show()

                plt.figure()
                plt.scatter(mutual_info_15, initial_recall_error)
                plt.xlabel('Mutual Information of Input Patterns at 15 Patterns')
                plt.ylabel('Initial Recall Error')
                plt.show()

                plt.figure()
                plt.scatter(mutual_info_15, final_recall_error)
                plt.xlabel('Mutual Information of Input Patterns at 15 Patterns')
                plt.ylabel('Final Recall Error')
                plt.show()

                for jdx in range(len(pre_breakpoint_weight_matrix)):
                    w_jdx = pre_breakpoint_weight_matrix[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(w_jdx, aspect='auto', interpolation='none')
                    plt.xlabel('Output Cell ID #')
                    plt.ylabel('Input Cell ID #')
                    plt.title('Pre-Breakpoint Weight Matrix')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()

                for jdx in range(len(breakpoint_weight_matrix)):
                    w_jdx = breakpoint_weight_matrix[jdx]
                    ax = plt.figure().gca()
                    plt.imshow(w_jdx, aspect='auto', interpolation='none')
                    plt.xlabel('Output Cell ID #')
                    plt.ylabel('Input Cell ID #')
                    plt.title('Post-Breakpoint Weight Matrix')
                    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                    ax.set_xticks(np.arange(w_jdx.shape[1]+1)-.5, minor=True)
                    ax.set_yticks(np.arange(w_jdx.shape[0]+1)-.5, minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    ax.tick_params(which='minor', bottom=False, left=False)
                    plt.show()




                # for jdx in range(len(all_input_matrices)):
                #     p_jdx = all_input_matrices[jdx]
                #     ax = plt.figure().gca()
                #     plt.imshow(p_jdx, aspect='auto', interpolation='none')
                #     plt.ylabel('Input Cell ID #')
                #     plt.xlabel('Input Pattern #')
                #     plt.title('Breakpoint Patterns')
                #     ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
                #     ax.set_xticks(np.arange(p_jdx.shape[1]+1)-.5, minor=True)
                #     ax.set_yticks(np.arange(p_jdx.shape[0]+1)-.5, minor=True)
                #     ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                #     ax.tick_params(which='minor', bottom=False, left=False)
                #     print (jdx, breakpoint_pattern_set)
                #     ax.set_xlim([breakpoint_pattern_set[jdx]-0.5,breakpoint_pattern_set[jdx]+10.5])
                #     plt.show()

                # for jdx in range(len(all_overlap_traces)):
                #     overlap_trial = all_overlap_traces[jdx]
                #     plt.figure()
                #     # print('length check', NPATT, overlap_tracker_T[0])
                #     for a in range(NCELL):
                #         overlap_trace = overlap_trial[a]
                #         plt.plot(overlap_trace)
                #     plt.xlabel('Number of Patterns')
                #     plt.ylabel('Number of counts')
                #     plt.xlim(breakpoint_overlap[jdx], breakpoint_overlap[jdx]+10)
                #     plt.title('Breakpoint Cell Usage')
                #     plt.show()

                # add_new_pattern(p, w, NCELL, SPATT, ACTIVATION_THRESHOLD)
                # average_percent_error, average_crosspattern_activation, max_error, min_error = calculate_errors(NPATT, ACTIVATION_THRESHOLD, w, p)
                # print("Storing {} patterns yields an average {}%% error".format(NPATT, average_percent_error))
                # print("The average cross-pattern activation for spurious firing was {}".format(average_crosspattern_activation))

                ERROR_LIST.append(average_percent_error)
                
                # if average_percent_error - ERROR_THRESHOLD > 0:
                #     break
                np.savetxt(saveFolder+FWGT, w, delimiter=' ', fmt='%d')
                np.savetxt(saveFolder+FPATT, p, delimiter=' ', fmt='%d')
            
            # print("Finished for NCELL:{} SPATT:{} Max NPATT:{}".format(NCELL, SPATT_RATIO, NPATT))

            # plot_error(NPATT_LIST, ERROR_LIST)
            error_diff = [abs(_ - ERROR_THRESHOLD) for _ in ERROR_LIST]
            min_val, min_index = min((val,idx) for (idx, val) in enumerate(error_diff))
            max_capacity = NPATT_LIST[min_index]
            capacity_heatmap[jj][ii] = max_capacity
            jj = jj + 1
        ii = ii + 1
    return capacity_heatmap

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

    if sum(output_pattern) == 0:
        print("WARNING: No output activity")
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

def calculate_errors(NPATT, ACTIVATION_THRESHOLD, w, p):
    all_total_errors = []
    all_crosspattern_activation = []
    for CUE in range(NPATT):
        active_input_cells = np.where(p[:,CUE] == 1)[0]
        activation_count = np.zeros(len(w[0]), dtype=int)
        for output_cell in range(0, len(w[0])):
            for active_input in active_input_cells:
                if w[output_cell][active_input].any() == 1:
                    activation_count[output_cell] = activation_count[output_cell] + 1

        # print(activation_count)
        output_pattern = activation_count > ACTIVATION_THRESHOLD
        if sum(output_pattern) == 0:
            print("WARNING: No output activity")
        spurious_error = 0
        deleterious_error = 0
        for i in range(0, len(output_pattern)):
            if output_pattern[i].any() == 1 and p[i][CUE].any() == 0:
                spurious_error = spurious_error + 1
                all_crosspattern_activation.append(activation_count[i])                
            elif output_pattern[i].any() == 0 and p[i][CUE].any() == 1:
                deleterious_error = deleterious_error + 1
        total_error = spurious_error + deleterious_error
        all_total_errors.append(total_error)

    max_error = max(all_total_errors)
    min_error = min(all_total_errors)
    # if NPATT == 1:
    #     print(max_error, min_error)

    average_total_error = sum(all_total_errors)/len(all_total_errors)
    average_percent_error = average_total_error*100/len(w[0])
    if len(all_crosspattern_activation) != 0:
        average_crosspattern_activation = sum(all_crosspattern_activation)/len(all_crosspattern_activation)
    else:
        average_crosspattern_activation = 0
    return average_percent_error, average_crosspattern_activation, max_error, min_error

def plot_error(NPATT_LIST, ERROR_LIST):
    ERROR_THRESHOLD_LINE = [ERROR_THRESHOLD for _ in ERROR_LIST]
    
    plt.figure()
    plt.plot(NPATT_LIST, ERROR_LIST)
    plt.plot(NPATT_LIST, ERROR_THRESHOLD_LINE, linestyle='--', color='tab:gray')
    plt.title("Average Recall Error for Increasing Number of Patterns")
    plt.xlabel("Number of Patterns Stored")
    plt.ylabel("Average Total Error (%)")
    plt.xticks(np.arange(0,100+10,10))
    plt.show()

capacity_heatmap = generate_patterns(NCELL_LIST, NPATT_LIST, SPATT_LIST)

plt.figure()
plt.imshow(capacity_heatmap, vmin=0, extent=[min(NCELL_LIST),max(NCELL_LIST), max(SPATT_LIST), min(SPATT_LIST)], aspect='auto')
ax = plt.gca()
ax.invert_yaxis()
plt.colorbar()
plt.title("Storage capacity depending on Network Properties")
plt.xlabel("Network Size")
plt.ylabel("Pattern Sparsity")
plt.show()

# fig, ax = plt.subplots()
# _, _, bar_container = ax.hist(capacity_heatmap, HIST_BINS, lw=1)
# ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 50, repeat=False, blit=True)
# plt.show()
