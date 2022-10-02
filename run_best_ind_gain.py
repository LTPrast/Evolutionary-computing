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

######################## SET-UP FRAMEWORK AND FUNCTIONS ###########################

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e

def compare_algorithms(experiment_name_1, experiment_name_2, enemy):
    """
    Comparing two experiments with diffeent EAs
    
    experiment_name_1 = name of first experiment to find directory
    experiment_name_2 = name of second experiment to find directory
    enemy = the opponent for both experiments
    labels = list of labels for legend in right order i.e. 1 then 2
    """
    experiments = [experiment_name_1, experiment_name_2]
    mean_fitness = []
    max_fitness = []
    labels_mean = []
    labels_max = []
    
    # For each sigma read file and append the dataframe
    for experiment in experiments:
        max_fitness_cur = pd.read_csv(f'./{experiment}/{experiment}_max_fitness.csv',delimiter=",")
        max_fitness.append(max_fitness_cur)
    
        # label for line in the plot
        ylabel = experiment[:len(experiment) - 13]
        labels_max.append('max fitness '+ylabel)
    
        mean_fitness_cur = pd.read_csv(f'./{experiment}/{experiment}_mean_fitness.csv',delimiter=",")
        mean_fitness.append(mean_fitness_cur)
    
        # label for line in the plot
        labels_mean.append('mean fitness '+ylabel)
    
    number_of_generations = len(max_fitness[0].values[0]) - 1
    number_of_trials = len(max_fitness[0].values)
    
    # Define plot colours
    colour = ['blue', 'red', 'dodgerblue', 'orange']
    
    fig, ax = plt.subplots(1, figsize=(10,8))
    # fig, ax = plt.subplots(2, figsize=(10,8))
    
    # Do for the different parameters
    for i in range(len(mean_fitness)):
        
        # Create lists
        average_max_fitness = []
        std_max_fitness = []
        average_mean_fitness = []
        std_mean_fitness =[]
        
        # Iterate over the generation and add the mean and standard deviation of the 10 runs to a list
        for j in range(1, number_of_generations + 1):
            generation = "Generation_"+str(j)
            
            average_max_fitness.append(np.mean(max_fitness[i][generation]))
            std_max_fitness.append(np.std(max_fitness[i][generation]))
            
            average_mean_fitness.append(np.mean(mean_fitness[i][generation]))
            std_mean_fitness.append(np.std(mean_fitness[i][generation]))
    
        generations = np.arange(1, number_of_generations+1, 1)
        average_max_fitness = np.array(average_max_fitness)
        std_max_fitness = np.array(std_max_fitness)
        average_mean_fitness = np.array(average_mean_fitness)
        std_mean_fitness = np.array(std_mean_fitness)
    
        # Print the max and mean of the final generation
        print(experiments[i] + ' after ' + str(number_of_generations) + ' generations:' )
        print('max = ', average_max_fitness[-1])
        print('mean = ', average_mean_fitness[-1])
    
        # Plot fitness lines
        ax.plot(generations, average_max_fitness, color=colour[i], label=labels_max[i])
        ax.fill_between(generations, average_max_fitness-std_max_fitness, average_max_fitness+std_max_fitness, alpha=0.2, edgecolor=colour[i], facecolor=colour[i])
        ax.plot(generations, average_mean_fitness, color=colour[i+2], label=labels_mean[i])
        ax.fill_between(generations, average_mean_fitness-std_mean_fitness, average_mean_fitness+std_mean_fitness, alpha=0.2, edgecolor=colour[i+2], facecolor=colour[i+2])
        
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticks([0,20,40,60,80,100])
        ax.set_ylabel(f"Fitness", fontsize=20)

    # Plot and save
    plt.xlabel("Generation",fontsize=20)
    plt.title(f"EA Comparison for Enemy {enemy}", fontsize=20)
    plt.savefig(f'./{experiments[0]}/EA_comparison_fitness_enemy_{enemy}.jpg', dpi=300)
    return


####################### RUN ONCE ###########################

opponents = [[1], [4], [6]]

algos = ['basic', 'island']

fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True)
c = 0

# Choose which parameters to plot
params = 'original'

for opponent in opponents:
    # Get the data
    for algo in algos:

        experiment_name = f'{algo}_algo_enemy_{opponent[0]}'

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

        if params == 'new':
            best_solutions = pd.read_csv(f'./{experiment_name}/new_params_{algo}_algo_enemy_{opponent[0]}_highest_gain.csv',delimiter=",")
            best_solutions_extra = pd.read_csv(f'./{experiment_name}/new_params_{algo}_algo_enemy_{opponent[0]}_highest_gain.csv',delimiter=",")
        else:
            best_solutions = pd.read_csv(f'./{experiment_name}/{experiment_name}_highest_gain.csv',delimiter=",")
            best_solutions_extra = pd.read_csv(f'./{experiment_name}_extra_trials/{experiment_name}_extra_trials_highest_gain.csv',delimiter=",")
        best_solutions_extra = best_solutions_extra.to_numpy()
        best_solutions = best_solutions.to_numpy()
        print(best_solutions.shape)
        best_solutions = np.concatenate((best_solutions, best_solutions_extra))
        print(best_solutions.shape)
        
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
    
    # Get plots
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

# Save plot
ax[0].set_ylabel("Individual Gain", fontsize=18)
plt.xticks([1,2],['Basic', 'Island'])
plt.yticks([-20,0,20,40,60,80,100])
plt.savefig(f'./EA_comparison_boxplots.jpg', dpi=300)

########################### PLOT MAX AND MEAN PLOT ###################################

for opponent in opponents:
    for algo in algos:
        compare_algorithms(f'{algos[0]}_algo_enemy_{opponent[0]}', f'{algos[1]}_algo_enemy_{opponent[0]}', opponent[0])