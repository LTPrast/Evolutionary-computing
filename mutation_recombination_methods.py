# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:21:17 2022

@author: arong
"""

import numpy as np

################### Recombination Methods ###################################

def simple_arithmetic_recombination(parent_1, parent_2):
        
    # pick random crossover point
    k = np.random.randint(0, len(parent_1))
    
    # find average of parents after point k
    part_2 = np.mean( np.array([ parent_1[k:], parent_2[k:] ]), axis=0 )
    
    child_1 = np.append(parent_1[:k], part_2)
    child_2 = np.append(parent_2[:k], part_2)
    
    return child_1, child_2 




################### Mutation Methods ########################################