# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:06:23 2022

@author: arong
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


name_1 = 'TSP'
name_2 = 'sigma'

TSPs = [10, 50, 50, 30, 10,]
sigmas = [0.2, 0.1, 0.05, 0.225, 0.225,]


mean_fitness = []
max_fitness = []
labels_mean = []
labels_max = []
   
# For each sigma read file and append the dataframe
for i in range(len(TSPs)):
    
   TSP = TSPs[i]
   sigma = sigmas[i]
   max_fitness_cur = pd.read_csv(f'./TSP_{TSP}_sigma_{sigma}/TSP_{TSP}_sigma_{sigma}_max_fitness.csv',delimiter=",")
   max_fitness.append(max_fitness_cur)
   
   # label for line in the plot
   labels_max.append('t-size = '+  f' {TSP}, '+r'$\sigma$ ='+f'{sigma}')
   
   mean_fitness_cur = pd.read_csv(f'./TSP_{TSP}_sigma_{sigma}/TSP_{TSP}_sigma_{sigma}_mean_fitness.csv',delimiter=",")
   mean_fitness.append(mean_fitness_cur)
   
   # label for line in the plot
   labels_mean.append('t-size = '+  f' {TSP}, '+r'$\sigma$ ='+f'{sigma}')
   
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
plt.title("Tuning; Mean Fitness", fontsize=18)
plt.savefig('tuning_mean_fitness_algorithm_1.pdf',bbox_inches='tight')


# max fitness = 89.7604602175716 for TSP = 30 and sigma =0.2

# max fitness = 76.89451668706867 for TSP = 10 and sigma =0.05
# max fitness = 77.85939908426535 for TSP = 50 and sigma =0.125


# mean fitness = 80.59560736995077 for TSP = 40 and sigma =0.075
# mean fitness = 82.21888380497008 for TSP = 30 and sigma =0.2


# mean fitness = 66.18140458846543 for TSP = 10 and sigma =0.15
# mean fitness = 66.6830530466882 for TSP = 10 and sigma =0.25