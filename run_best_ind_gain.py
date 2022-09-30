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

import glob, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy.stats as st

# import evolutionary algorithm functions
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


####################### RUN ONCE ###########################

opponents = [[1], [4], [6]]
algos = ['basic', 'island']

fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True)
c = 0

for opponent in opponents:
    for algo in algos:

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

        # save value of runs
        with open(experiment_name+'/'+experiment_name+'_ind_gains', "wb") as file:   #Pickling
            pickle.dump(ind_gains, file)
    
    
    # GET PLOTS
    experiments = [f'{algos[0]}_algo_enemy_{opponent[0]}', f'{algos[1]}_algo_enemy_{opponent[0]}']
    
    # Define plot colours (lightblue and lightred)
    colour = [(0.2, 0.5, 1), (1, 0.5, 0.5)]

    gains = []

    for i in range(2):
        with open(f'./{experiments[i]}/{experiments[i]}_ind_gains', "rb") as file:
            ind_gains = pickle.load(file)
            print(ind_gains)
            gains.append(ind_gains)

        box = ax[c].boxplot(ind_gains, positions=[i+1], patch_artist=True, medianprops=dict(color='black'))
        plt.setp(box["boxes"], facecolor=colour[i])
    
    _, p = st.ttest_ind(gains[0], gains[1])
    print('T test p-value:', p)

    _, p = st.ttest_ind(gains[0], gains[1], equal_var=False)
    print('Welch test p-value:', p)

    # Plot all
    ax[c].tick_params(axis='both', which='major', labelsize=18)
    ax[c].set_title("Enemy %i" % (opponent[0]), fontsize=18)

    c += 1

ax[0].set_ylabel("Individual Gain", fontsize=18)
plt.xticks([1,2],['Basic', 'Island'])
plt.yticks([-20,0,20,40,60,80,100])
plt.savefig(f'./EA_comparison_boxplots.jpg', dpi=300)