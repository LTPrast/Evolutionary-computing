# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:06:26 2022

@author: liza
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
import pickle

# import evolutionary algorithm functions
from parent_selection_methods import *
from mutation_recombination_methods import *
from survival_methods import *
from plotting_functions import *

######################### SET-UP FRAMEWORK ###################################

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e

# evaluates fitness of every individual in population
# def evaluate(x):
#     fit = []
#     player_hp = []
#     enemy_hp = []
#     for agent in x:
#         f, p, e = simulation(env, agent)
#         fit.append(f)
#         player_hp.append(p)
#         enemy_hp.append(e)
#     return np.array(fit), np.array(player_hp), np.array(enemy_hp)


####################### SET EXPERIMENT PARAMETERS ###########################

opponents = [[4]]
algos = ['basic', 'island']

for opponent in opponents:
    for algo in algos:
        
        if opponent[0] == 4:
            np.random.seed(100)
        else:
            np.random.seed(99)

        experiment_name = f'{algo}_algo_enemy_{opponent[0]}'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                        enemies=opponent,
                        playermode="ai",
                        player_controller=player_controller(10),
                        enemymode="static",
                        level=2,
                        speed="fastest")

        # genetic algorithm params
        run_mode = 'test' # train or test

            
        #################### PERFORM EXPERIMENT #####################################

        start_time = time.time()

        # create empty arrays to store values for every experiment
        ind_gains = []

        #best_solutions = pd.read_csv(f'./{experiment_name}/{experiment_name}_highest_gain.csv',delimiter=",")
        best_solutions = pd.read_csv(f'./{experiment_name}/{algo}_algo_enemy_{opponent[0]}_highest_gain.csv',delimiter=",")
        best_solutions = best_solutions.to_numpy()
        
        # number of times for this experiment to be repeated
        nr_trials = best_solutions.shape[0]

        for i in range(nr_trials):

            gains = []
            
            agent_nn = np.array(best_solutions[i][1:],dtype=np.float32)
            f, p, e = simulation(env,agent_nn)
            gains.append(p - e)
            
            ind_gains.append(np.mean(gains))

        end_time = time.time()

        execution_time = ((end_time-start_time)/60)/60
        print("Runtime was ", execution_time, " hours")

        # save value of runs
        with open(experiment_name+'/'+experiment_name+'_ind_gains', "wb") as file:   #Pickling
            pickle.dump(ind_gains, file)

    comp_algos_boxplots(f'{algos[0]}_algo_enemy_{opponent[0]}', f'{algos[1]}_algo_enemy_{opponent[0]}', opponent[0])

    