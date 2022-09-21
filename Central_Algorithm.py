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
tournament_size = 10                    # Number of individuals taking part in tournamnet selection 
sigma = 0.1                             # gene mutation probability 

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
experiment_name = 'TSP_'+str(tournament_size)+'_sigma_'+str(sigma)
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


dom_u = 1                               # Max weight for neural network
dom_l = -1                              # Min weight for neural network
dist_std = 0.1                          # mean of distribution to draw sizes for gene mutation
npop = 100                              # Population size
gens = 10                               # number of generations
individuals_deleted = 30                # number of individuals killed every generation
num_offspring = individuals_deleted     # equal number of offspring to keep constant population size

    
#################### PREFORM EXPERIMENT #####################################

start_time = time.time()

# number of times for this experiment to be repeated
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
    
        # empty matrix for children such that they don't take part in 
        # reproduction of this cycle
        children = np.empty((0,n_vars), float)
        
        # create offspring in 2 ways i.e. pure mutation and recombination
        #followed by mutation
        for i in range(int(num_offspring/2)):
            
            parent_1 = tournament_selection(population, fit_pop, tournament_size)
            parent_2 = tournament_selection(population, fit_pop, tournament_size)
            
            if i % 2 == 0:

                child_1, child_2 = simple_arithmetic_recombination(parent_1, parent_2)
                child_1 = gaussian_mutation(child_1, sigma, dist_std)
                child_2 = gaussian_mutation(child_2, sigma, dist_std)
            
            else: 

                child_1 = gaussian_mutation(parent_1, sigma, dist_std)
                child_2 = gaussian_mutation(parent_2, sigma, dist_std)
                
            
            # append each child to children array
            children = np.append(children, np.array([child_1]), axis=0)
            children = np.append(children, np.array([child_2]), axis=0)
    
        # append all offspring of this generation to population
        population = np.vstack([population, children])
        
        # evaluate population
        fit_pop = evaluate(population)
        
        # kill certain number of worst individuals
        population, fit_pop = kill__x_individuals(population, fit_pop, individuals_deleted)
        
        # evaluate whole population and store values
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

