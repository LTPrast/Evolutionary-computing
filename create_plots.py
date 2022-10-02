from plotting_functions import *

name_1 = 'tsize'
name_2 = 'sigma'

tsizes = [10, 20, 30, 40, 50]
sigmas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]

# compare a set of trials for different paramter values, plot max and mean on seperate plot
# also prints a sorted list from worst to best
tuning_plot_mean_fitness(tsizes, sigmas, name_1, name_2)
tuning_plot_max_fitness(tsizes, sigmas, name_1, name_2)

# 3D plot 
tuning_3D_trisurface_plot_max_fitness(tsizes, sigmas, name_1, name_2)

# compare two algorithms, max and mean with std
compare_algorithms('TSP_30_sigma_0.2' , 'TSP_10_sigma_0.05')