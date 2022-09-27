# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:19:23 2022

@author: arong
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

def tuning_plot_mean_fitness(parameter_1, parameter_2, name_1, name_2):
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
        plt.plot(generations, average_mean_fitness, linestyle='dashed' , label=labels_mean[i])
        
    
    # Plot all
    plt.legend(fontsize=20)
    plt.xlabel("Generation", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title("Tuning; Mean Fitness", fontsize=20)
    plt.show()
    
    print("sorted list from worst to best measured by mean fitness after ", number_of_generations, " generations")
    index_sort = np.argsort(np.array(final_values))  
    for index in index_sort:
        print('mean fitness = '+ str(final_values[index]) + ' for '+name_1+' = '+str(parameters[index][0])+' and '+name_2+' =' +str(parameters[index][1]))
    
    return
    

def tuning_plot_max_fitness(parameter_1, parameter_2, name_1, name_2):
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
        plt.plot(generations, average_max_fitness, linestyle='dashed', label=labels_max[i])
        
    
    # Plot all
    plt.legend(fontsize=20)
    plt.xlabel("Generation", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title("Tuning; Max Fitness", fontsize=20)
    plt.show()
    
    print("sorted list from worst to best measured by max fitness after ", number_of_generations, " generations")
    index_sort = np.argsort(np.array(final_values))  
    for index in index_sort:
        print('max fitness = '+ str(final_values[index]) + ' for '+name_1+' = '+str(parameters[index][0])+' and '+name_2+' =' +str(parameters[index][1]))
    
    return


def compare_algorithms(experiment_name_1, experiment_name_2):
    """
    Comparing two experiments with diffeent EAs
    
    experiment_name_1 = name of first experiment to find directory
    experiment_name_2 = name of second experiment to find directory
    labels = list of labels for legend in right order i.e. 1 then 2
    """
    experiments = [experiment_name_1, experiment_name_2]
    mean_fitness = []
    max_fitness = []
    labels_mean = []
    labels_max = []
    
    # For each sigma read file and append the dataframe
    for experiment in experiments:
        max_fitness_cur = pd.read_csv(f'./{experiment}/{experiment}_max_fitness.csv',delimiter=",")
        max_fitness.append(max_fitness_cur)
    
        # label for line in the plot
        labels_max.append('max fitness '+f' {experiment}')
    
        mean_fitness_cur = pd.read_csv(f'./{experiment}/{experiment}_mean_fitness.csv',delimiter=",")
        mean_fitness.append(mean_fitness_cur)
    
        # label for line in the plot
        labels_mean.append('mean fitness '+f' {experiment}')
    
    number_of_generations = len(max_fitness[0].values[0]) - 1
    number_of_trials = len(max_fitness[0].values)
    
    # Define plot colours
    colour = ['blue', 'red']
    
    # Do for the different parameters
    for i in range(len(mean_fitness)):
        
        # Create lists
        average_max_fitness = []
        std_max_fitness = []
        average_mean_fitness = []
        std_mean_fitness =[]
        
        # Iterate over the generation and add the mean and standard deviation of the 10 runs to a list
        for j in range(1, number_of_generations + 1):
            generation = "Generation_"+str(j)
            
            average_max_fitness.append(np.mean(max_fitness[i][generation]))
            std_max_fitness.append(np.std(max_fitness[i][generation]))
            
            average_mean_fitness.append(np.mean(mean_fitness[i][generation]))
            std_mean_fitness.append(np.std(mean_fitness[i][generation]))
    
        generations = np.arange(1, number_of_generations+1, 1)
        average_max_fitness = np.array(average_max_fitness)
        std_max_fitness = np.array(std_max_fitness)
        average_mean_fitness = np.array(average_mean_fitness)
        std_mean_fitness = np.array(std_mean_fitness)
    
        # Print the max and mean of the final generation
        print(experiments[i] + ' after ' + str(number_of_generations) + ' generations:' )
        print('max = ', average_max_fitness[-1])
        print('mean = ', average_mean_fitness[-1])
    
        # Plot fitness lines
        plt.plot(generations, average_max_fitness, linestyle='dashed' ,color=colour[i], label=labels_max[i])
        plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
        plt.plot(generations, average_mean_fitness, linestyle ='dotted', color=colour[i], label=labels_mean[i])
        plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
        
    # Plot all
    plt.legend(fontsize=20)
    plt.xlabel("Generation",fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title("EA Comparisson", fontsize=20)
    plt.show()
    return


def tuning_3D_trisurface_plot_max_fitness(parameter_1, parameter_2, name_1, name_2):
    """
    parameter_1 = list of parameters used for tuning
    parameter_2 = list of second parameters used for tuning
    name_1 = name of parameter 1
    name_2 = name of parameter 2
    """
    
    max_fitness = []
    parameters = []
    
    # For each sigma read file and append the dataframe
    for par1 in parameter_1:
        
        for par2 in parameter_2:
            max_fitness_cur = pd.read_csv(f'./TSP_{par1}_sigma_{par2}/TSP_{par1}_sigma_{par2}_max_fitness.csv',delimiter=",")
            max_fitness.append(max_fitness_cur)
        
            # label for line in the plot
            parameters.append([par1, par2])
            
        number_of_generations = len(max_fitness[0].values[0]) - 1


    # Do for the different parameters
    x_axis = []
    y_axis = []
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
        x_axis.append(parameters[i][0])
        y_axis.append(parameters[i][1])

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(np.array(x_axis) , np.array(y_axis), np.array(final_values), linewidth=0, antialiased=True,cmap=cm.jet)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=20)
    ax.set_xlabel(name_1, fontsize=20, rotation=150)
    ax.set_ylabel(name_2, fontsize=20)
    ax.set_zlabel('Max Fitness', fontsize=20, rotation=60)
    return


