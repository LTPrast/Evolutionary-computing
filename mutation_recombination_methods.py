import numpy as np

################### Recombination Methods ###################################

def simple_arithmetic_recombination(parent_1, parent_2):
    """
    This function takes two individuals as indputs and return two offspring
    created using simple arithmetic combination. A random point k is chosen in
    the length of the genome. The first child consists of the first parents 
    genome up to point k, after point k it is the average of the genome of both
    parents. Child 2 up to point k consits of the genome of parent 2 while after
    point k it is again the average genome of both parents.
    """
        
    # pick random crossover point
    k = np.random.randint(0, len(parent_1))
    
    # find average of parents after point k
    part_2 = np.mean( np.array([ parent_1[k:], parent_2[k:] ]), axis=0 )
    
    child_1 = np.append(parent_1[:k], part_2)
    child_2 = np.append(parent_2[:k], part_2)
    
    return child_1, child_2 

def one_point_crossover(parent_1, parent_2):
    """
    1-point recombination by exchanging tails (2nd parts)
    Get a random index and swap the 2nd parts between parents after that index.
    Point selection: exclude 1st and last indices to avoid offspring being clones of parents.
    """
    # get random point (exclude edges)
    point = np.random.randint(1, len(parent_1) - 1)
    
    # swap tails
    child_1 = np.concatenate((parent_1[:point], parent_2[point:]))
    child_2 = np.concatenate((parent_2[:point], parent_1[point:]))
    
    return child_1, child_2

################### Mutation Methods ########################################

def mutation(parent, sigma, std, dist='gaussian'):
    """
    This function takes a parent and a positionwise mutation probability sigma
    as inputs and returns a single offspring.
    
    This function can also be used to add mutation to an offspring which has
    undergone recombination.
    """
    child = np.zeros(len(parent))
    
    for i in range(len(child)):       
        if np.random.random() < sigma:
            old_gene = parent[i]

            if dist == 'gaussian':
                mutation = np.random.normal(0, std)
            else:
                mutation = np.random.uniform(-std, std)

            new_gene = old_gene + mutation
            
            if -1 < new_gene < 1: 
                child[i] = new_gene
            else:
                new_gene = old_gene - mutation
                child[i] = new_gene

        else:
            child[i] = parent[i]
            
    return child


def reset_or_creep_mutation(parent, resetting_sigma=0.001, creep_sigma=0.2, creep_std=0.1):
    """
    2 types of mutation: low probability to reset the weight to a random value
    from a (-1, 1) uniform distribution (random resetting)
    or just add to the weight a value taken from a N(0, 0.1) distribution (creep mutation)
    """

    child = np.zeros(len(parent))
    
    # for each weight of the individual apply one of the two mutations
    for i in range(len(child)):       
        # random resetting with mutation_rate probability
        if np.random.uniform(0,1) <= resetting_sigma:
            child[i] = np.random.uniform(-1, 1)
        # creep mutation
        elif np.random.uniform(0,1) <= creep_sigma:
            child[i] = parent[i] + np.random.normal(0, creep_std)
        else:
            child[i] = parent[i]

    return child