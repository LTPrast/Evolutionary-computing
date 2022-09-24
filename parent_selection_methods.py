# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:19:04 2022

@author: arong
"""

import numpy as np

def tournament_selection(population, fit_pop, k):
    """
    This function preforms a tournament selection of the population, the inputs
    are the population, the fitness of the population and the tournament size.
    
    The function return the winner of the tournament.
    """
    # pick random index of population and set current winner to this index
    max_idx = len(fit_pop)
    parent_idx = np.random.randint(0, max_idx)

    # perform k direct comparissons to random members of population, if one has
    # a higher fitness than the current member, update the current winner index.
    for _ in range(k):
        rnd_idx = np.random.randint(0, max_idx)

        if fit_pop[rnd_idx] > fit_pop[parent_idx]:
            parent_idx = rnd_idx
            
    # return the parent according to the winning index
    parent = population[parent_idx][:]

    return parent


def fittest_with_random_parents(pop, fit_pop):
    """
    select two individuals for recombination
    parent_1: the one with the highest fitness
    parent_2: a random individual
    """
    # get the fittest parent_1
    i_1 = np.argmax(fit_pop) # parent_1 index
    parent_1 = pop[i_1]
    fit_1 = fit_pop[i_1]
    # get a random parent_2
    i_2 = np.random.randint(len(fit_pop)) # parent_2 index
    parent_2 = pop[i_2]
    fit_2 = fit_pop[i_2]
    return parent_1, parent_2, fit_1, fit_2

