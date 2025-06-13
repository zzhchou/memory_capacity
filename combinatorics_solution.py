from itertools import combinations, permutations
from itertools import product

def generate_index_distributions(S, B):
    """
    Generate all possible ways to distribute S indices among B blocks,
    considering how many indices each block gets using integer compositions.
    """
    def compositions(n, k):
        """Generate all k-part compositions of n."""
        if k == 1:
            yield [n]
        else:
            for i in range(n + 1):
                for rest in compositions(n - i, k - 1):
                    yield [i] + rest
    
    print((compositions(S,B)))
    return set(compositions(S, B))

def generate_combinations(S, B):
    """
    For each distribution, generate all possible index assignments.
    """
    elements = list(range(S))  # Indices from 0 to S-1
    distributions = generate_index_distributions(S, B)
    
    all_assignments = {}
    total_combinations = 0
    for distribution in distributions:
        block_indices = list(product(*(combinations(elements, size) for size in distribution)))
        for block_index in block_indices:
            print ('block_index', block_index)
            for block in block_index:
                print('block', block)
                if block != ():
                    for idx in range(len(block)):
                        print('block[idx]', block[idx], idx, S)
                        block[idx] = block[idx]+idx*S
        all_assignments[distribution] = block_indices
        total_combinations += len(block_indices)
    
    return all_assignments, total_combinations

# Example usage
S = 3  # Indices per partition
B = 3  # Number of blocks

assignments, total_combinations = generate_combinations(S, B)
for dist, combos in assignments.items():
    print(f"Distribution {dist}: {len(combos)} combinations")
    for combo in combos:
        print(combo)

print(f"Total number of combinations: {total_combinations}")
