import numpy as np

def tournament_selection(population, fit_pop, k):
    """
    This function preforms a tournament selection of the population, the inputs
    are the population, the fitness of the population and the tournamnet size.
    
    The function return the winner of the tournamanet.
    """

    # pick random index of population and set current winner to this indes
    max_idx = len(fit_pop)
    parent_idx = np.random.randint(0, max_idx)

    # preform k direct comparissons to random members of population, if one has
    # a higher fitness than the current member, update the current winner index.
    for i in range(k):
        rnd_idx = np.random.randint(0, max_idx)

        if fit_pop[rnd_idx] > fit_pop[parent_idx]:
            parent_idx = rnd_idx
            
    # return the parent according to the winning index
    parent = population[parent_idx][:]

    return parent