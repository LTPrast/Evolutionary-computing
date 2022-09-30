# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:14:25 2022

@author: arong
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from parent_selection_methods import *
from mutation_recombination_methods import *
from survival_methods import *


def island_mutations(population, fit_pop, gain, num_sub_pop, num_offspring, tournament_size, dist_std, sigma, evaluate):
    """
    population = population of all islands
    fit_pop = fitness of all individulas of all islands
    num_sub_pop = number of sub population i.e. islands 
    num_offspring = total number of offspring across all islands
    tournament_size = tournament size on each island
    dist_std = standard deviation of chosen distribution
    sigma = mutation probability 
    evaluate = evaluation function
    """
    
    offspring_per_island = int(num_offspring/num_sub_pop)
    index_increment = int(len(fit_pop)/num_sub_pop)
    start = 0
    stop = index_increment
    
    # iterate through all subpopulations
    for x in range(num_sub_pop):
        
        sub_pop = population[start:stop][:]
        fit_sub_pop = fit_pop[start:stop]
        gain_sub_pop = gain[start:stop]
        # create offspring for sub-population
        for y in range(int(offspring_per_island/2)):
            
            # select parents of subpopulation
            parent_1 = tournament_selection(sub_pop, fit_sub_pop, tournament_size)
            parent_2 = tournament_selection(sub_pop, fit_sub_pop, tournament_size)
            
            if y % 2 == 0:
                
                child_1, child_2 = simple_arithmetic_recombination(parent_1, parent_2)
                child_1 = gaussian_mutation(child_1, sigma, dist_std)
                child_2 = gaussian_mutation(child_2, sigma, dist_std)
            
            else:
                child_1 = gaussian_mutation(parent_1, sigma, dist_std)
                child_2 = gaussian_mutation(parent_2, sigma, dist_std)
            
            # evaluate each child
            child_1_fitness, php1, ehp1 = evaluate([child_1])
            child_2_fitness, php2, ehp2 = evaluate([child_2])
            child_1_gain = php1 - ehp1
            child_2_gain = php2 - ehp2
            #child_1_fitness = 0
            #child_2_fitness = 0
            #for i in range(5):
            #    child_1_fitness += evaluate([child_1])[0]/5
            #    child_2_fitness += evaluate([child_2])[0]/5
            
            # find worst indidivudals of subpopulation
            index_sorted = np.argsort(fit_sub_pop)
            # find local (subpopulation level) and global (population level) index of worst member of subpopulation
            worst_idx_local = index_sorted[0]
            worst_idx_global = worst_idx_local + start
            # find local (subpopulation level) and global (population level) index of 2nd worst member of subpopulation
            worst2_idx_local = index_sorted[1]
            worst2_idx_global = worst2_idx_local + start
            
            # replace worst member of subpopulation if child has higher fitness
            if child_1_fitness > fit_pop[worst_idx_global]:
                population[worst_idx_global][:] = child_1
                fit_pop[worst_idx_global] = child_1_fitness
                gain[worst_idx_global] = child_1_gain

                if child_2_fitness > fit_pop[worst2_idx_global]:
                    population[worst2_idx_global][:] = child_2
                    fit_pop[worst2_idx_global] = child_2_fitness
                    gain[worst2_idx_global] = child_2_gain

            elif child_2_fitness > fit_pop[worst_idx_global]:
                population[worst_idx_global][:] = child_2
                fit_pop[worst_idx_global] = child_2_fitness
                gain[worst_idx_global] = child_2_gain

        start += index_increment
        stop += index_increment
    
    return population, fit_pop, gain


def island_migration(population, fit_pop, num_sub_pop, tournament_size, migration_magnitude):
    
    index_increment = int(len(fit_pop)/num_sub_pop)
    stop = index_increment
    
    starting_islands = np.arange(0, num_sub_pop, 1)
    destination_islands = np.arange(0, num_sub_pop, 1)
    np.random.shuffle(destination_islands)

    for x in range(num_sub_pop):
        
        # check that starting point and destination are not the same
        if starting_islands[x] != destination_islands[x]:
            
            start_island_start = starting_islands[x]*index_increment
            start_island_stop = start_island_start + index_increment
            
            destination_island_start = destination_islands[x]*index_increment
            destination_island_stop = destination_island_start + index_increment
            
            start_island_pop = population[start_island_start:start_island_stop][:]
            start_island_fit = fit_pop[start_island_start:start_island_stop]
            
            destination_island_pop = population[destination_island_start:destination_island_stop][:]
            destination_island_fit = fit_pop[destination_island_start:destination_island_stop]

            
            for y in range(migration_magnitude):
                
                # find migrant using tournament selection
                migrant, migrant_idx = tournament_selection_migration(start_island_pop, start_island_fit, tournament_size)
                migrant_fitness = start_island_fit[migrant_idx]
                
                # find worst individual in destination island
                index_sorted = np.argsort(destination_island_fit)
                worst_individual_local_index = index_sorted[0]

                worst_individual_global_index = destination_island_start + worst_individual_local_index
                
                
                population[worst_individual_global_index][:] = migrant
                fit_pop[worst_individual_global_index] = migrant_fitness
    
    return population, fit_pop
            

def tournament_selection_migration(population, fit_pop, k):
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

    return parent, parent_idx
                

def plot_sub_populations(sub_plot_data, num_sub_pop, gens, trial_num,  max_fit=True):
    
    index_increment = int(len(sub_plot_data[0][:])/num_sub_pop)
    
    mean_fitness = np.zeros((num_sub_pop, gens))
    max_fitness = np.zeros((num_sub_pop, gens))
    
    for i in range(gens): # row
        start_idx = 0
        stop_idx = index_increment
        for j in range(num_sub_pop):
            fitness_current_generation = sub_plot_data[i][:]
            mean_fitness[j][i] = np.mean(fitness_current_generation[start_idx:stop_idx])
            max_fitness[j][i] = np.max(fitness_current_generation[start_idx:stop_idx])
            start_idx += index_increment
            stop_idx += index_increment
    
    generations = np.arange(1, gens + 1, 1)
    colour = ['lightskyblue', 'blue','pink', 'orange', 'red']
    
    if max_fit:
    
        for x in range(num_sub_pop):

            plt.plot(generations, max_fitness[x], linestyle='dotted' ,color=colour[x], label='max sub-population '+ str(x))
            plt.title('Max Fitness Iteration '+str(trial_num))
            
    else: 
        for x in range(num_sub_pop):
            plt.plot(generations, mean_fitness[x], linestyle='dashed' ,color=colour[x], label='mean sub-population '+ str(x))
            plt.title('Mean Fitness Iteration '+str(trial_num))

    plt.legend()
    plt.show()
    
    return
        
    
    