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
                  enemies=[7],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


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

# kills worst x% of population
def kill_worst_x_percent(population, fit_pop, kill_perc):
    fraction = int(len(fit_pop)*kill_perc)
    
    for i in range(fraction):
        # indicies sorted from worst to best solution
        index_sorted = np.argsort(fit_pop)
        # index of worst solution
        index = index_sorted[0]
        
        # delete that solution and it's fitness
        population = np.delete(population, index ,0)
        fit_pop = np.delete(fit_pop, index)
    
    return population, fit_pop

def tournament_selection(population, fit_pop, k):
    # pick random index of population
    max_idx = len(fit_pop)
    parent_idx = np.random.randint(0, max_idx)
    
    for i in range(k):
        rnd_idx = np.random.randint(0, max_idx)
        if fit_pop[rnd_idx] > fit_pop[parent_idx]:
            parent_idx = rnd_idx
    
    parent = population[parent_idx][:]
    
    return parent

# create 2 offspring from 2 parents
def simple_arithmetic_recombination(parent_1, parent_2):
        
    # pick random crossover point
    k = np.random.randint(0, len(parent_1))
    
    # find average of parents after point k
    part_2 = np.mean( np.array([ parent_1[k:], parent_2[k:] ]), axis=0 )
    
    child_1 = np.append(parent_1[:k], part_2)
    child_2 = np.append(parent_2[:k], part_2)
    
    return child_1, child_2 


    
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