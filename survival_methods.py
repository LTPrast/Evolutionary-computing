# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:24:15 2022

@author: arong
"""

import numpy as np

def kill_worst_x_percent(population, fit_pop, kill_perc):

    # number of individuals to be deleted
    num_to_kill = int(len(fit_pop)*kill_perc)

    for i in range(num_to_kill):
        # indicies sorted from worst to best solution
        index_sorted = np.argsort(fit_pop)
        # index of worst solution
        index = index_sorted[0]
        
        # delete that solution and it's fitness
        population = np.delete(population, index ,0)
        fit_pop = np.delete(fit_pop, index)
        
    return population, fit_pop



