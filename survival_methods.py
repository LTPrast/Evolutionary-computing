# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:24:15 2022

@author: arong
"""

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


def replace_only_with_fitter_offspring(pop, fit_pop, offspring, fit_off):
    """
    Offspring evaluation is required before this is used.
    Compare each child with the current lowest fitness individual and if the child
    wins replace the individual and its fitness. Repeat for all children.
    """
    # for all children
    for i in range(len(fit_off)):
        # if child has higher fitness, replace the worst individual and their fitness
        min_index = np.argmin(fit_pop)
        if fit_off[i] > fit_pop[min_index]:
            pop[min_index] = offspring[i]
            fit_pop[min_index] = fit_off[i]
    return pop, fit_pop



