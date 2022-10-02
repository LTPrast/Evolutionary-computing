import numpy as np

def kill__x_individuals(population, fit_pop, x):
    """
    this function takes the population, fitness of the population and number
    of individuals to be delted as inputs
    
    it returns the new population along with it's fitness values'
    """

    for i in range(x):
        # indicies sorted from worst to best solution
        index_sorted = np.argsort(fit_pop)
        # index of worst solution
        index = index_sorted[0]
        
        # delete that solution and it's fitness
        population = np.delete(population, index ,0)
        fit_pop = np.delete(fit_pop, index)
        
    return population, fit_pop