# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:42:38 2022

@author: arong
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


max_fitness_1 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_max_fitness.csv',delimiter=",")
mean_fitness_1 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_mean_fitness.csv',delimiter=",")
max_fitness_05 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform05_max_fitness.csv',delimiter=",")
mean_fitness_05 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform05_mean_fitness.csv',delimiter=",")
max_fitness_02 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform02_max_fitness.csv',delimiter=",")
mean_fitness_02 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform02_mean_fitness.csv',delimiter=",")

number_of_generations = len(max_fitness_1.values[0]) - 1
number_of_trials = len(max_fitness_1.values)


mean_fitness = [mean_fitness_02, mean_fitness_05, mean_fitness_1]
max_fitness = [max_fitness_02, max_fitness_05, max_fitness_1]
colour = ['blue', 'red', 'black']
labels_mean = ['0.2 mean', '0.5 mean', 'random new mean']
labels_max = ['0.2 max', '0.5 max', 'random new max']

for i in range(3):
    
    average_max_fitness = []
    std_max_fitness = []
    average_mean_fitness = []
    std_mean_fitness =[]
    
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
    std_mean_fitness =np.array(std_mean_fitness)


    plt.plot(generations, average_max_fitness, linestyle='dashed' ,color=colour[i], label=labels_max[i])
    plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    plt.plot(generations, average_mean_fitness, color=colour[i], label=labels_mean[i])
    plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Tuning Uniform Mutation Parameter")
plt.show()