# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:36:26 2022

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

######################### SET-UP FRAMEWORK ###################################

TSP_perc = 0.15         # Percentage of population taking part in tournamnet selection (as decimal)
sigma = 0.1             # gene mutation probability 

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
experiment_name = 'mutation_probability_'+str(sigma)+'_tournament_size_'+str(TSP_perc)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluates fitness of every individual in population
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


####################### SET EXPERIMENT PARAMETERS ###########################

np.random.seed(99)

n_hidden_neurons = 10

opponents = [4]
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


dom_u = 1               # Max weight for neural network
dom_l = -1              # Min weight for neural network
npop = 100              # Population size
gens = 10               # number of generations
kill_perc = 0.25        # percentage of population killed every generation (as decimal)
offspring_perc = 0.25   # percentage of offspring every generation (as decimal)
    
#################### PREFORM EXPERIMENT #####################################

start_time = time.time()

experiment_iterations = 10

# create empty arrays to store values for each generation if every experiment 
# iteration
average_fitness_data = np.empty((0, gens+1), float)
max_fitness_data = np.empty((0,gens+1), float)
fitness_std_data = np.empty((0,gens+1), float)
best_solution_data = np.empty((0,n_vars), float)

for iteration in range(experiment_iterations):

    # Create Populations, one individual is one row in a matrix
    population = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    
        
    # find fitness of each member of the initial population
    fit_pop = evaluate(population)
    # store population size to later analyze if remaining constant
    pop_size = [len(fit_pop)]
    # find the index of the fitest individual
    best_solution_index = np.argmax(fit_pop)
    # find the fitness of the best solution
    fitness_of_best_solution = [fit_pop[best_solution_index]]
    # find the mean fitness of the population
    mean = [np.mean(fit_pop)]
    # find the standard deviation in fitness of the population
    std = [np.std(fit_pop)]
    
    for iteration in range(gens):
    
        # kill worst part of population
        population, fit_pop = kill_worst_x_percent(population, fit_pop, kill_perc)
    
        # find the number of offspring to be generated
        num_offspring = int(npop*offspring_perc)
    
        # empty matrix for children such that they don't take part in 
        # reproduction of this cycle
        children = np.empty((0,n_vars), float)
        
        # create offspring in 2 ways i.e. pure mutation and recombination
        #followed by mutation
        for i in range(int(num_offspring/2)):
            
            # find tournament size based on TSP_perc
            k = int(TSP_perc*len(fit_pop))
            parent_1 = tournament_selection(population, fit_pop, k)
            parent_2 = tournament_selection(population, fit_pop, k)
            
            if i % 2 == 0:

                child_1, child_2 = simple_arithmetic_recombination(parent_1, parent_2)
                child_1 = uniform_mutation(child_1, sigma)
                child_2 = uniform_mutation(child_2, sigma)
            
            else: 

                child_1 = uniform_mutation(parent_1, sigma)
                child_2 = uniform_mutation(parent_2, sigma)
                
            
            # append each child to children array
            children = np.append(children, np.array([child_1]), axis=0)
            children = np.append(children, np.array([child_2]), axis=0)
    
        # append all offspring of this generation to population
        population = np.vstack([population, children])
    
        
        # evaluate whole population and store values
        fit_pop = evaluate(population)
        pop_size.append(len(fit_pop))
        best_solution_index = np.argmax(fit_pop)
        fitness_of_best_solution.append(fit_pop[best_solution_index])
        mean.append(np.mean(fit_pop))
        std.append(np.std(fit_pop))
        
    average_fitness_data = np.append(average_fitness_data, [np.array(mean)], axis=0)
    max_fitness_data = np.append(max_fitness_data,  [np.array(fitness_of_best_solution)], axis=0)
    fitness_std_data = np.append(fitness_std_data, [np.array(std)], axis=0)
    best_solution_data = np.append(best_solution_data, [np.array(population[best_solution_index][:])], axis=0)

    

end_time = time.time()

execution_time = ((end_time-start_time)/60)/60
print("Runtime was ", execution_time, " hours")

##################### Save Data in Files ####################################

# One file to store set-up

set_up_dict = {
  "Population_size": npop,
  "Generations": gens,
  "Killing_percentage": kill_perc,
  "Offspring_percentage": offspring_perc,
  "Tournamnet_selection_percentage": TSP_perc,
  "Gene_mutation_probability": sigma,
  "Enemies": opponents,
  "Level": difficulty,
  "Iterations:":experiment_iterations,
  "Experiment Name": experiment_name,
  "Execution Time": execution_time
}

file_name = experiment_name + '_set_up.csv'
set_up_df = pd.DataFrame.from_dict(set_up_dict)
set_up_df.to_csv(file_name)


# row is trial
# column is generation

columns = ['Generation_'+str(i+1) for i in range(gens+1)]
rows = ['Trial_'+str(i+1) for i in range(experiment_iterations)]
    

# One file for mean fitness
file_name = experiment_name + '_mean_fitness.csv'
mean_fitness_df = pd.DataFrame(average_fitness_data)
mean_fitness_df.columns = columns
mean_fitness_df.index = rows
mean_fitness_df.to_csv(file_name)

# one file for max fitness
file_name = experiment_name + '_max_fitness.csv'
max_fitness_df = pd.DataFrame(max_fitness_data)
max_fitness_df.columns = columns
max_fitness_df.index = rows
max_fitness_df.to_csv(file_name)

# One file for standard deviation
file_name = experiment_name + '_std_fitness.csv'
std_fitness_df = pd.DataFrame(fitness_std_data)
std_fitness_df.columns = columns
std_fitness_df.index = rows
std_fitness_df.to_csv(file_name)


# One file for best solution
rows = ['Trial_'+str(i+1) for i in range(experiment_iterations)]

file_name = experiment_name + '_best_solution.csv'
best_df = pd.DataFrame(best_solution_data)
best_df.index = rows
best_df.to_csv(file_name)

