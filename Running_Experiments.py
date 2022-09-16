# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:00:44 2022

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

from parent_selection_methods import *
from mutation_recombination_methods import *
from survival_methods import *

np.random.seed(10)

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1               # Max weight for neural network
dom_l = -1              # Min weight for neural network
npop = 100               # Population size
gens = 10                # number of generations
kill_perc = 0.25         # percentage of population killed every generation (as decimal)
offspring_perc = 0.25    # percentage of offspring every generation (as decimal)
TSP_perc = 0.07           # Percentage of population taking part in tournamnet selection (as decimal)


# Create Populations, one individual is one row in a matrix
population = np.random.uniform(dom_l, dom_u, (npop, n_vars))
# evaluation

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluates fitness of every individual in population
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

    
# store values for initial population
fit_pop = evaluate(population)  
pop_size = [len(fit_pop)]
best_solution = [np.argmax(fit_pop)]
fitness_of_best_solution = [fit_pop[best_solution][0]]
mean = [np.mean(fit_pop)]
std = [np.std(fit_pop)]

for iteration in range(gens):
    
    # kill worst part of population
    population, fit_pop = kill_worst_x_percent(population, fit_pop, kill_perc)
    
    # create offspring using tournamnet selection
    num_offspring = int(npop*offspring_perc)
    num_offspring = num_offspring if num_offspring % 2 == 0 else num_offspring + 1

    
    # create 1 random individual as a base for children matrix
    children = np.zeros(n_vars)
    
    # create num_offspring offspring 
    for i in range(int(num_offspring/2)):
        
        # find parents using tournamnet selection
        k = int(TSP_perc*len(fit_pop))
        parent_1 = tournament_selection(population, fit_pop, k)
        parent_2 = tournament_selection(population, fit_pop, k)
        
        # create children using simple arithmetic recombination
        child_1, child_2 = simple_arithmetic_recombination(parent_1, parent_2)
        
        # add children to population
        children = np.vstack([children, child_1])
        children = np.vstack([children, child_2])
    
    population = np.vstack([population, children[1:][:]])

    
    # evaluate whole population and store values
    fit_pop = evaluate(population)
    pop_size.append(len(fit_pop))
    new_best_solution = np.argmax(fit_pop)
    best_solution.append(new_best_solution)
    fitness_of_best_solution.append(fit_pop[new_best_solution])
    mean.append(np.mean(fit_pop))
    std.append(np.std(fit_pop))

    
iterations = np.arange(0, gens+1, 1)

best_solution = np.array(best_solution)
fitness_of_best_solution = np.array(fitness_of_best_solution)
mean = np.array(mean)
std = np.array(std)

plt.plot(iterations, fitness_of_best_solution, color='red', label='Best Solution')
plt.plot(iterations, mean, color='blue', label='Mean Solution')
plt.fill_between(iterations, mean-std, mean+std, alpha=0.2, edgecolor='blue', facecolor='blue')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("fitness")
plt.show()