# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:42:38 2022

@author: arong
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


mean_fitness_1 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_mean_fitness.csv',delimiter=",")
mean_fitness_05 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform05_mean_fitness.csv',delimiter=",")
mean_fitness_02 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_uniform02_mean_fitness.csv',delimiter=",")
mean_gaussian_005 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.05_mean_fitness.csv',delimiter=",")
mean_gaussian_01 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.1_mean_fitness.csv',delimiter=",")
mean_gaussian_02 = pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.2_mean_fitness.csv',delimiter=",")
mean_gaussian_05 =  pd.read_csv('./mutation_probability_0.1_tournament_size_10_gaussian_0.5_mean_fitness.csv',delimiter=",")

number_of_generations = len(mean_fitness_1.values[0]) - 1
number_of_trials = len(mean_fitness_1.values)


mean_fitness = [mean_fitness_02, mean_fitness_05, mean_fitness_1, mean_gaussian_005, mean_gaussian_01, mean_gaussian_02, mean_gaussian_05]
colour = ['lightskyblue', 'blue', 'black','pink', 'orange', 'red', 'firebrick']
labels_mean = ['uniform std=0.2', 'uniform std=0.5', 'random new gene','gaussian std=0.05','gaussian std=0.1', 'gaussian std=0.2', 'gaussian std=0.5']

for i in range(len(mean_fitness)):
    
    average_mean_fitness = []
    std_mean_fitness =[]
    
    for j in range(1, number_of_generations + 1):
        generation = "Generation_"+str(j)
        
        average_mean_fitness.append(np.mean(mean_fitness[i][generation]))
        std_mean_fitness.append(np.std(mean_fitness[i][generation]))
    

    generations = np.arange(1, number_of_generations+1, 1)
    average_mean_fitness = np.array(average_mean_fitness)
    std_mean_fitness =np.array(std_mean_fitness)

    plt.plot(generations, average_mean_fitness, linestyle='dotted', color=colour[i], label=labels_mean[i])
    # plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
    
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Average Fitness; Tuning Uniform Mutation Parameter")
plt.show()