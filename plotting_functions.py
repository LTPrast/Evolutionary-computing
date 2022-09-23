# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:19:23 2022

@author: arong
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def tuning_plot_mean_fitness(parameter_1, parameter_2, name_1, name_2, colour, std=False):
    """
    parameter_1 = list of parameters used for tuning
    parameter_2 = list of second parameters used for tuning
    name_1 = name of parameter 1
    name_2 = name of parameter 2
    colours = list of colours to be used
    std = if std shall be plotted
    """
    
    mean_fitness = []
    labels_mean = []
    parameters = []
    
    # For each sigma read file and append the dataframe
    for par1 in parameter_1:
        
        for par2 in parameter_2:
            mean_fitness_cur = pd.read_csv(f'./TSP_{par1}_sigma_{par2}/TSP_{par1}_sigma_{par2}_mean_fitness.csv',delimiter=",")
            mean_fitness.append(mean_fitness_cur)
        
            # label for line in the plot
            labels_mean.append(name_1 + '= ' f'{par1}, ' + name_2 +' = ' + f' {par2}')
            parameters.append([par1, par2])
            
        number_of_generations = len(mean_fitness[0].values[0]) - 1
        number_of_trials = len(mean_fitness[0].values)
    

    # Do for the different parameters
    final_values = []
    for i in range(len(mean_fitness)):
        
        # Create lists
        average_mean_fitness = []
        std_mean_fitness = []
        max_value = 0
        
        # Iterate over the generation and add the mean and standard deviation of the 10 runs to a list
        for j in range(1, number_of_generations + 1):
            generation = "Generation_"+str(j)
            
            average_mean_fitness.append(np.mean(mean_fitness[i][generation]))
            std_mean_fitness.append(np.std(mean_fitness[i][generation]))
            
    
        generations = np.arange(1, number_of_generations+1, 1)
        average_mean_fitness = np.array(average_mean_fitness)
        std_mean_fitness = np.array(std_mean_fitness)
        
        final_values.append(average_mean_fitness[-1])

        # Plot fitness lines
        plt.plot(generations, average_mean_fitness, linestyle='dashed' ,color=colour[i], label=labels_mean[i])
        
        if std:
            plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    
    # Plot all
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Tuning; Mean Fitness")
    plt.show()
    
    print("sorted list from worst to best measured by mean fitness after ", number_of_generations, " generations")
    index_sort = np.argsort(np.array(final_values))  
    for index in index_sort:
        print('mean fitness = '+ str(final_values[index]) + ' for '+name_1+' = '+str(parameters[index][0])+' and '+name_2+' =' +str(parameters[index][1]))
    
    return
    

def tuning_plot_max_fitness(parameter_1, parameter_2, name_1, name_2, colour, std=False):
    """
    parameter_1 = list of parameters used for tuning
    parameter_2 = list of second parameters used for tuning
    name_1 = name of parameter 1
    name_2 = name of parameter 2
    colours = list of colours to be used
    std = if std shall be plotted
    """
    
    max_fitness = []
    labels_max = []
    parameters = []
    
    # For each sigma read file and append the dataframe
    for par1 in parameter_1:
        
        for par2 in parameter_2:
            max_fitness_cur = pd.read_csv(f'./TSP_{par1}_sigma_{par2}/TSP_{par1}_sigma_{par2}_max_fitness.csv',delimiter=",")
            max_fitness.append(max_fitness_cur)
        
            # label for line in the plot
            labels_max.append(name_1 + '= ' f'{par1}, ' + name_2 +' = ' + f' {par2}')
            parameters.append([par1, par2])
            
        number_of_generations = len(max_fitness[0].values[0]) - 1
        number_of_trials = len(max_fitness[0].values)
    

    # Do for the different parameters
    final_values = []
    for i in range(len(max_fitness)):
        
        # Create lists
        average_max_fitness = []
        std_max_fitness = []
        max_value = 0
        
        # Iterate over the generation and add the mean and standard deviation of the 10 runs to a list
        for j in range(1, number_of_generations + 1):
            generation = "Generation_"+str(j)
            
            average_max_fitness.append(np.mean(max_fitness[i][generation]))
            std_max_fitness.append(np.std(max_fitness[i][generation]))
            
    
        generations = np.arange(1, number_of_generations+1, 1)
        average_max_fitness = np.array(average_max_fitness)
        std_max_fitness = np.array(std_max_fitness)
        
        final_values.append(average_max_fitness[-1])

        # Plot fitness lines
        plt.plot(generations, average_max_fitness, linestyle='dashed' ,color=colour[i], label=labels_max[i])
        
        if std:
            plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    
    # Plot all
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Tuning; Max Fitness")
    plt.show()
    
    print("sorted list from worst to best measured by max fitness after ", number_of_generations, " generations")
    index_sort = np.argsort(np.array(final_values))  
    for index in index_sort:
        print('max fitness = '+ str(final_values[index]) + ' for '+name_1+' = '+str(parameters[index][0])+' and '+name_2+' =' +str(parameters[index][1]))
    
    return


colour = ['blue', 'red', 'green', 'orange']
name_1 = 'TSP'
name_2 = 'sigma'

TSPs = [30]
sigmas = [0.05, 0.1, 0.15, 0.2]

tuning_plot_mean_fitness(TSPs, sigmas, name_1, name_2, colour, std=False)
tuning_plot_max_fitness(TSPs, sigmas, name_1, name_2, colour, std=False)