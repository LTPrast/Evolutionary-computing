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

def uniform_mutation(parent, sigma):
    """
    This function takes a parent and a positionwise mutation probability sigma
    as inputs and returns a single offspring.
    """
    child = np.zeros(len(parent))
    
    for i in range(len(child)):
        
        if np.random.random() < sigma:
            new_gene = np.random.uniform(-1,1)
            child[i] = new_gene
            
        else:
            child[i] = parent[i]
            
    return child

def self_adaptive_mutation(parent):
    child = np.zeros(len(parent))
    # add to current value drawn from distribution
    # adapt sigma value of distribution
    return child
    



        