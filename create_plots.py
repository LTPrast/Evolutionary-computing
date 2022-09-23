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

tuning_plot_mean_fitness(TSPs, sigmas, name_1, name_2, colour, std=False)
tuning_plot_max_fitness(TSPs, sigmas, name_1, name_2, colour, std=False)
