# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:36:26 2022

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
    return f

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

opponents = [[1],[4],[6]]
algos = ['normal', 'island']

for opponent in opponents:
    for algo in algos:
        
        np.random.seed(99)

        experiment_name = f'enemy_{opponent[0]}_{algo}'
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

        # number of times for this experiment to be repeated
        nr_trails = 10

        # create empty arrays to store values for every experiment
        ind_gains = np.empty((0,10), float)

        best_solutions = pd.read_csv(f'./{experiment_name}/{experiment_name}_highest_gain.csv',delimiter=",")

        for i in range(nr_trails):

            gains = []
            for j in range(5):
                gains.append(evaluate(best_solutions[i]))
            
            ind_gains.append(np.mean(gains))

        end_time = time.time()

        execution_time = ((end_time-start_time)/60)/60
        print("Runtime was ", execution_time, " hours")

        ##################### Save Data in File ####################################

        def save_file(data, file_name, experiment_name, cols=True, rows=True):
            # Save the data into a dataframe and save in csv file
            df = pd.DataFrame(data)

            # if cols == True:
            #     columns = ['Generation_'+str(i+1) for i in range(gens+1)]
            #     df.columns = columns
            # if rows == True:
            #     rows = ['Trial_'+str(i+1) for i in range(experiment_iterations)]
            #     df.index = rows
            df.to_csv(experiment_name+'/'+experiment_name+file_name)
            return

        # save value of runs
        save_file(ind_gains, '_ind_gains.csv', experiment_name)

    comp_algos_boxplots(f'enemy_{opponent[0]}_{algos[0]}', f'enemy_{opponent[0]}_{algos[1]}')

    