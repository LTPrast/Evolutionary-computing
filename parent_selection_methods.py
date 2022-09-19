# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:19:04 2022

@author: arong
"""

import numpy as np

def tournament_selection(population, fit_pop, k):
    # pick random index of population
    max_idx = len(fit_pop)
    parent_idx = np.random.randint(0, max_idx)

    for i in range(k):
        rnd_idx = np.random.randint(0, max_idx)

        if fit_pop[rnd_idx] > fit_pop[parent_idx]:
            parent_idx = rnd_idx

    parent = population[parent_idx][:]

    return parent

