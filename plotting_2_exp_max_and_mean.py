# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:42:38 2022

@author: arong
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TOURNAMENT SIZE
TSP = 30

sigmas = [0.05, 0.1, 0.15, 0.2]
mean_fitness = []
max_fitness = []
labels_mean = []
labels_max = []

# For each sigma read file and append the dataframe
for sigma in sigmas:
    max_fitness_cur = pd.read_csv(f'./TSP_{TSP}_sigma_{sigma}/TSP_{TSP}_sigma_{sigma}_max_fitness.csv',delimiter=",")
    max_fitness.append(max_fitness_cur)

    # label for line in the plot
    labels_max.append(r'max for $\sigma$ ='+f' {sigma}')

    mean_fitness_cur = pd.read_csv(f'./TSP_{TSP}_sigma_{sigma}/TSP_{TSP}_sigma_{sigma}_mean_fitness.csv',delimiter=",")
    mean_fitness.append(mean_fitness_cur)

    # label for line in the plot
    labels_mean.append(r'mean for $\sigma$ ='+f' {sigma}')

number_of_generations = len(max_fitness[0].values[0]) - 1
number_of_trials = len(max_fitness[0].values)

# Define plot colours
colour = ['blue', 'red', 'green', 'orange']

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
    print('sigma =', sigmas[i])
    print('max:', average_max_fitness[-1])
    print('mean:', average_mean_fitness[-1])

    # Plot fitness lines
    plt.plot(generations, average_max_fitness, linestyle='dashed' ,color=colour[i], label=labels_max[i])
    plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    plt.plot(generations, average_mean_fitness, color=colour[i], label=labels_mean[i])
    plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    
# Plot all
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Tuning Mutation Parameter")
plt.show()