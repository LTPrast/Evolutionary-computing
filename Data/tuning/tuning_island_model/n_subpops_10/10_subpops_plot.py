# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:06:23 2022

@author: arong
"""

from plotting_functions import *


name_1 = 'migration_rate'
name_2 = 'migration_magnitude'

migration_rates = [5, 10, 15, 20]
migration_magnitudes = [1, 2, 3, 4,5]

# compare a set of trials for different paramter values, plot max and mean on seperate plot
# also prints a sorted list from worst to best
# tuning_plot_mean_fitness(migration_rates, migration_magnitudes, name_1, name_2)
tuning_plot_max_fitness(migration_rates, migration_magnitudes, name_1, name_2)


# 3D plot 
# tuning_3D_trisurface_plot_max_fitness(migration_rates, migration_magnitudes, name_1, name_2)

# compare two algorithms, max and mean with std
# compare_algorithms('TSP_30_sigma_0.2' , 'TSP_10_sigma_0.05')

# mean fitness = 89.5163934897187 for migration_rate = 5 and migration_magnitude =5
# mean fitness = 90.74903557547937 for migration_rate = 5 and migration_magnitude =4

# max fitness = 91.07362709743956 for migration_rate = 5 and migration_magnitude = 4