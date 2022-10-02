# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:42:38 2022

@author: arong
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


max_fitness_1 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_max_fitness.csv',delimiter=",")
max_fitness_05 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform05_max_fitness.csv',delimiter=",")
max_fitness_02 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform02_max_fitness.csv',delimiter=",")
max_gaussian_005 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.05_max_fitness.csv',delimiter=",")
max_gaussian_01 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.1_max_fitness.csv',delimiter=",")
max_gaussian_02 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.2_max_fitness.csv',delimiter=",")
max_gaussian_05 =  pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.5_max_fitness.csv',delimiter=",")


number_of_generations = len(max_fitness_1.values[0]) - 1
number_of_trials = len(max_fitness_1.values)


max_fitness = [max_fitness_02, max_fitness_05, max_fitness_1, max_gaussian_005, max_gaussian_01, max_gaussian_02, max_gaussian_05]
colour = ['lightskyblue', 'blue', 'black','pink', 'orange', 'red', 'firebrick']
labels_max = ['uniform std=0.2', 'uniform std=0.5', 'random new gene','gaussian std=0.05','gaussian std=0.1', 'gaussian std=0.2', 'gaussian std=0.5']

for i in range(len(max_fitness)):
    
    average_max_fitness = []
    std_max_fitness = []

    
    for j in range(1, number_of_generations + 1):
        generation = "Generation_"+str(j)
        
        average_max_fitness.append(np.mean(max_fitness[i][generation]))
        std_max_fitness.append(np.std(max_fitness[i][generation]))
        
    
    generations = np.arange(1, number_of_generations+1, 1)
    average_max_fitness = np.array(average_max_fitness)
    std_max_fitness = np.array(std_max_fitness)


    plt.plot(generations, average_max_fitness, linestyle='dashed' ,color=colour[i], label=labels_max[i])
    # plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])   

plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Maximum Fitness; Tuning Uniform Mutation Parameter")
plt.show()