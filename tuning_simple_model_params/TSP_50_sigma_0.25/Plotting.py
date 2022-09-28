# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:42:38 2022

@author: arong
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


max_fitness = pd.read_csv('TSP_50_sigma_0.25_max_fitness.csv', delimiter=",")
mean_fitness = pd.read_csv('TSP_50_sigma_0.25_mean_fitness.csv',delimiter=",")

number_of_generations = len(max_fitness.values[0]) - 1
number_of_trials = len(max_fitness.values)

average_max_fitness = []
std_max_fitness = []
average_mean_fitness = []
std_mean_fitness =[]

for j in range(1, number_of_generations + 1):
    generation = "Generation_"+str(j)
    
    average_max_fitness.append(np.mean(max_fitness[generation]))
    std_max_fitness.append(np.std(max_fitness[generation]))
    
    average_mean_fitness.append(np.mean(mean_fitness[generation]))
    std_mean_fitness.append(np.std(mean_fitness[generation]))
    
    

generations = np.arange(1, number_of_generations+1, 1)
average_max_fitness = np.array(average_max_fitness)
std_max_fitness = np.array(std_max_fitness)
average_mean_fitness = np.array(average_mean_fitness)
std_mean_fitness =np.array(std_mean_fitness)


plt.plot(generations, average_max_fitness, color='red', label='Best Solution')
plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(generations, average_mean_fitness, color='blue', label='Mean Solution')
plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor='blue', facecolor='blue')
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()

print(average_max_fitness)
print(average_mean_fitness)