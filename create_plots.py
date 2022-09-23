# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:06:23 2022

@author: arong
"""

from plotting_functions import *


colour = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'black', 'olive', 'cyan', 'yellow', 'navy', 'darkviolet', 'lime', 'firebrick', 'gray', 'peru', 'darkkhaki', 'slateblue', 'lightcoral']
name_1 = 'TSP'
name_2 = 'sigma'

TSPs = [10, 30]
sigmas = [0.05, 0.1, 0.15, 0.2]

# compare a set of trials for different paramter values, plot max and mean on seperate plot
# also prints a sorted list from worst to best
tuning_plot_mean_fitness(TSPs, sigmas, name_1, name_2, colour, std=False)
tuning_plot_max_fitness(TSPs, sigmas, name_1, name_2, colour, std=False)


# 3D plot 
tuning_3D_trisurface_plot_max_fitnesss(TSPs, sigmas, name_1, name_2)

# compare two algorithms, max and mean with std
compare_algorithms('TSP_30_sigma_0.2' , 'TSP_10_sigma_0.05')
