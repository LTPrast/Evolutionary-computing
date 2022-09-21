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

def random_mutation(parent, sigma):
    """
    This function takes a parent and a positionwise mutation probability sigma
    as inputs and returns a single offspring.
    
    This function can also be used to add mutation to an offspring which has
    undergone recombination.
    """
    child = np.zeros(len(parent))
    
    for i in range(len(child)):
        
        if np.random.random() < sigma:
            new_gene = np.random.uniform(-0.5,0.5)
            child[i] = new_gene
            
        else:
            child[i] = parent[i]
            
    return child


def uniform_mutation(parent, sigma, std):
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


def gaussian_mutation(parent, sigma, std):
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
            mutation = np.random.normal(0, std)
            new_gene = old_gene + mutation
            
            if -1 < new_gene < 1: 
                child[i] = new_gene
            else:
                new_gene = old_gene - mutation
                child[i] = new_gene
            
        else:
            child[i] = parent[i]
            
    return child




        