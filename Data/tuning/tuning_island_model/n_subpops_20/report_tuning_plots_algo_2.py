# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:42:06 2022

@author: arong
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


name_1 = 'TSP'
name_2 = 'sigma'

rates = [5, 5, 20, 5, 20]
magnitudes = [5, 1, 4, 4, 1]


mean_fitness = []
max_fitness = []
labels_mean = []
labels_max = []
   
# For each sigma read file and append the dataframe
for i in range(len(rates)):
    
   rate = rates[i]
   magnitude = magnitudes[i]
   max_fitness_cur = pd.read_csv(f'./migration_rate_{rate}_migration_magnitude_{magnitude}/migration_rate_{rate}_migration_magnitude_{magnitude}_max_fitness.csv',delimiter=",")
   max_fitness.append(max_fitness_cur)
   
   # label for line in the plot
   labels_max.append('rate = '+  f' {rate}, '+'magnitude ='+f'{magnitude}')
   
   mean_fitness_cur = pd.read_csv(f'./migration_rate_{rate}_migration_magnitude_{magnitude}/migration_rate_{rate}_migration_magnitude_{magnitude}_mean_fitness.csv',delimiter=",")
   mean_fitness.append(mean_fitness_cur)
   
   # label for line in the plot
   labels_mean.append('rate = '+  f' {rate}, '+'magnitude ='+f'{magnitude}')
   
   number_of_generations = len(max_fitness[0].values[0]) - 1
   number_of_trials = len(max_fitness[0].values)
   
   # Define plot colours
   colour = ['blue', 'red', 'green', 'black', 'orange']
   
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

       
       average_max_fitness.append(float(np.mean(max_fitness[i][generation])))
       std_max_fitness.append(float(np.std(max_fitness[i][generation])))
       
       average_mean_fitness.append(float(np.mean(mean_fitness[i][generation])))
       std_mean_fitness.append(float(np.std(mean_fitness[i][generation])))
       
       

   generations = np.arange(1, number_of_generations+1, 1)
   average_max_fitness = np.array(average_max_fitness)
   std_max_fitness = np.array(std_max_fitness)
   average_mean_fitness = np.array(average_mean_fitness)
   std_mean_fitness = np.array(std_mean_fitness)
       
   # Plot fitness lines
   # plt.plot(generations, average_max_fitness, linestyle='dashed' ,color=colour[i], label=labels_max[i])
   # plt.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
   plt.plot(generations, average_mean_fitness, linestyle ='dotted', color=colour[i], label=labels_mean[i])
   # plt.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
   
# Plot all
plt.legend(fontsize=15)
plt.xlabel("Generation",fontsize=18)
plt.ylabel("Fitness", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.title("Tuning Island Model; Mean Fitness", fontsize=18)
plt.savefig('tuning_mean_fitness_algorithm_2.pdf',bbox_inches='tight')
