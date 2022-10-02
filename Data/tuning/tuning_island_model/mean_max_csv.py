import pandas as pd
import numpy as np

# Print function
def print_mean_and_max(folder, experiment, print=False):
    """
    Print the mean and maximum fitness at the last generation
    Comparing two experiments with diffeent EAs
    """
    
    # Read mean and max fitness files and make it a dataframe
    max_fitness = pd.read_csv(f'{folder}/{experiment}/{experiment}_max_fitness.csv',delimiter=",")
    mean_fitness = pd.read_csv(f'{folder}/{experiment}/{experiment}_mean_fitness.csv',delimiter=",")

    # Get the number of generations
    number_of_generations = len(max_fitness.values[0]) - 1

    # Get the mean and standard deviation of the mean and max fitness for the last generation
    generation = "Generation_"+str(number_of_generations)
    average_max_fitness = np.mean(max_fitness[generation])
    std_max_fitness = np.std(max_fitness[generation])
    average_mean_fitness = np.mean(mean_fitness[generation])
    std_mean_fitness = np.std(mean_fitness[generation])

    if print == True:
        # Print the max and mean of the final generation
        print(experiment + ' after ' + str(number_of_generations) + ' generations:' )
        print('max = ', average_max_fitness)
        print('mean = ', average_mean_fitness)
        print('max std = ', std_max_fitness)
        print('mean std = ', std_mean_fitness)

    return experiment, average_mean_fitness, average_max_fitness


# get mean and max fitness of a run
exps = []; avg_mean_fits = []; avg_max_fits = []

# DEFINE FOLDER NAME
folder = 'n_subpops_20'

for j in range(1,6):
    for i in [5,10,15,20]:
        exp, avg_mean_fit, avg_max_fit = print_mean_and_max(folder, f'migration_rate_{i}_migration_magnitude_{j}')
        exps.append(exp)
        avg_mean_fits.append(str(avg_mean_fit).replace(".", ","))
        avg_max_fits.append(str(avg_max_fit).replace(".", ","))

results = f"{folder}_mean_max"
df = pd.DataFrame([avg_mean_fits, avg_max_fits], ['mean', 'max'], exps)
df.to_csv(folder+'/'+results+'.csv', sep=';')