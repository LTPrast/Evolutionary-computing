# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:05:26 2022

@author: arong
"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import matplotlib.pyplot as plt
import pandas as pd

# import evolutionary algorithm functions
from parent_selection_methods import *
from mutation_recombination_methods import *
from survival_methods import *
from island_functions import *

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

opponents = [4]

experiment_name = 'island_algo_enemy_'+str(opponents)[1]
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, p, e

# evaluates fitness of every individual in population
def evaluate(x):
    fit = []
    player_hp = []
    enemy_hp = []
    for agent in x:
        f, p, e = simulation(env, agent)
        fit.append(f)
        player_hp.append(p)
        enemy_hp.append(e)
    return np.array(fit), np.array(player_hp), np.array(enemy_hp)


####################### SET EXPERIMENT PARAMETERS ###########################

np.random.seed(100)

n_hidden_neurons = 10
difficulty = 2
# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=opponents,
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                enemymode="static",
                level=difficulty,
                speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state


# genetic algorithm params
run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1                               # Max weight for neural network
dom_l = -1                              # Min weight for neural network
dist_std = 0.1                          # mean of distribution to draw sizes for gene mutation
npop = 200                              # Population size
num_sub_pop = 20                        # number of subpopulations
gens = 100                              # number of generations
individuals_deleted = 40                # number of individuals killed every generation
num_offspring = individuals_deleted     # equal number of offspring to keep constant population size
tournament_size = int(round(npop/num_sub_pop * 0.25))# Number of individuals taking part in tournamnet selection 
sigma = 0.1                             # gene mutation probability 
migration_rate = 5                      # every how many generations communication occurs between subpopulation
migration_magnitude = 4                 # how many members cross over to a new island

#################### PREFORM EXPERIMENT #####################################

start_time = time.time()

# number of times for this experiment to be repeated
experiment_iterations = 10

average_fitness_data = np.empty((0, gens+1), float)
max_fitness_data = np.empty((0,gens+1), float)
best_solution_data = np.empty((0,n_vars), float)
highest_gain_data = np.empty((0, n_vars), float)

for iteration in range(experiment_iterations):

    # Create Populations, one individual is one row in a matrix
    population = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    
    # find fitness of each member of the initial population
    #fit_pop = np.zeros(npop, float)
    #player_hp = np.zeros(npop, float)
    #enemy_hp = np.zeros(npop, float)
    #for i in range(5):
    #    f, p, e = evaluate(population)
    #    fit_pop += f/5
    #    player_hp += p/5
    #    enemy_hp += e/5
    fit_pop, player_hp, enemy_hp = evaluate(population)
    gain = player_hp - enemy_hp
    subpop_plot_data = np.array([fit_pop])
    
    best_solution_index = np.argmax(fit_pop)
    highest_gain_index = np.argmax(gain)
    # find the fitness of the best solution
    fitness_of_best_solution = [fit_pop[best_solution_index]]
    highest_gain = [gain[highest_gain_index]]
    # find the mean fitness of the population
    mean = [np.mean(fit_pop)]
    
    for generation in range(gens):
        
        population, fit_pop, gain = island_mutations(population, fit_pop, gain, num_sub_pop, num_offspring, tournament_size, dist_std, sigma, evaluate)
        
        if generation != 0 and generation % migration_rate == 0:
            population, fit_pop = island_migration(population, fit_pop, num_sub_pop, tournament_size, migration_magnitude)
    
        subpop_plot_data = np.vstack((subpop_plot_data, fit_pop))
            
        best_solution_index = np.argmax(fit_pop)
        fitness_of_best_solution.append(fit_pop[best_solution_index])
        mean.append(np.mean(fit_pop))

        highest_gain_index = np.argmax(gain)
        highest_gain.append(gain[highest_gain_index])
    
    average_fitness_data = np.append(average_fitness_data, [np.array(mean)], axis=0)
    max_fitness_data = np.append(max_fitness_data,  [np.array(fitness_of_best_solution)], axis=0)
    best_solution_data = np.append(best_solution_data, [np.array(population[best_solution_index][:])], axis=0)
    highest_gain_data = np.append(highest_gain_data, [np.array(population[highest_gain_index][:])], axis=0)

end_time = time.time()

execution_time = ((end_time-start_time)/60)/60
print("Runtime was ", execution_time, " hours")

##################### Save Data in Files ####################################

def save_file(data, file_name, experiment_name, cols=True, rows=True):
    # Save the data into a dataframe and save in csv file
    df = pd.DataFrame(data)

    if cols == True:
        columns = ['Generation_'+str(i+1) for i in range(gens+1)]
        df.columns = columns
    if rows == True:
        rows = ['Trial_'+str(i+1) for i in range(experiment_iterations)]
        df.index = rows
    df.to_csv(experiment_name+'/'+experiment_name+file_name)
    return

# One file to store set-up

set_up_dict = {
"Population size": npop,
"Generations": gens,
"Killed per generation": individuals_deleted,
"Offspring number": num_offspring,
"Tournament size": tournament_size,
"Gene mutation probability": sigma,
"Enemies": opponents,
"Level": difficulty,
"Iterations:":experiment_iterations,
"Experiment Name": experiment_name,
"Execution Time": execution_time,
"Migration Rate": migration_rate,
"Migration Magnitude": migration_magnitude,
"Number of subpopulations": num_sub_pop
}

save_file(set_up_dict, '_set_up.csv', experiment_name, cols=False, rows=False)

# One file for mean fitness, max fitness, standard deviation and best solution
save_file(average_fitness_data, '_mean_fitness.csv', experiment_name)
save_file(max_fitness_data, '_max_fitness.csv', experiment_name)
save_file(best_solution_data, '_best_solution.csv', experiment_name, cols=False)
save_file(highest_gain_data, '_highest_gain.csv', experiment_name, cols=False)