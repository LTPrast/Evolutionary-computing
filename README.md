# Studying the Effects of Preserving Diversity in an Evolutionary Algorithm on Performance and Robustness

This repository was built to explore the effect of superimposing a spatial component onto a basic Evolutionary Algorithm to analyze how this effects the performance and robustness of the algorithm. The spatial component got introduced by splitting the population into subpopulations (islands) between which individuals migrate at a given frequency. The idea is that this allows different populations to converge to different solutions, thus maintaining diversity and exploring a larger section of the solution space. The central points of interest were the maximum and mean fitness values and how these progress through generations as well as how conistently an algorithm converges to satisfying solutions. An evolutionary algorithm was constructed to optimize the weight of nodes of a neural network with the aim of creating a specialist agent in the evoman framework and was tested on three of the eight potential enemies.

## Process
The approach to this project was to construct both the basic algorithm as well as the island model algorithm. Both algorithms have the same core to allow for a fair comparison. Parent selection is preformed using tournament selection, offspring are created using two methods, a simple mutation as well as simple arithmetic combination followed by a mutation and the population size is kept constant by getting rid of the indiviudlas with the worst fitness values.

Following this, both algorithms' parameters were tuned. First an attempt was made to identify the best form of mutating indiviudlas i.e. from which distribution and with which standard deviations samples should be drawn to mutate genes in an indiviudal genome. Secondly for the basic algorithm two parameters were tuned, namely the tournament size for parent selection as well as the mutation probability, the probability of mutating an individual gene during the mutation process. Given these values the island model was tuned for three further parameters: the number of subpopulations (islands), the migration frequency (how often individuals migrate) and the migration magnitude (how many individuals move between islands).   

## Repository Set-Up
This section will give a brief overview of the different files and folders in the repository.

### Main Algorithm Files
`player_controlller.py` is the neural network. <br>
`basic_algorithm.py` contains the basic evolutionary algorithm. <br>
`island_algorithm.py` contains the central algorithm for the island model. <br>

### Methods Files
`parent_selection_methods.py` contains functions for the selection of individuals to produce offspring. <br>
`survival_methods.py` contains the functions to maintain population size and hence delete a certain number of individuals. <br>
`mutation_recombination_methods.py` contains the functions by which offspring is created given the genome of the parents. <br>
`island_functions.py` contains the functions for the island model responsible for creating offspring and migration. <br>

### Analysis files
The results of the experiments are stored in csv files, the following files were used for analysis of these results: <br>
`plotting_functions.py` contains functions to plot results in different ways depending on the application. <br>
`create_plots.py` calls upon these functions to create plots. <br>
`mean_max_csv.py` provides csv files with the final mean and max fitness values of several experiments. <br>
`run_best_ind_gain.py` calls the function to plot the individual gain boxplots, runs significance tests and plots mean and max fitness over time for both EA's together. <br>

### Folders
The `evoman` folder contains the evoman framework which should not be altered. <br>
The `Data` folder contains all our tuning and final runs results. It has several subfolders: <br>
- the `Data/trial_island_model_10_runs` folder <br>
This folder contains the results of a trial run with the island model along with graphs plotting the development of the different islands to develop an intuition for the way the model works. <br>
- the `Data/tuning/tuning_mutation_distribution` folder <br>
This folder contains the results of experimentation with different distribution types (mainly unifrom and gaussian) from which mutations to individual nodes are drawn. Furthermore, it contains graphs of the outcomes. The result of these experiments was to use a gaussian distribution with mean and a standrad deviation of 0.1. <br>
- the `Data/tuning/tuning_simple_model_params` folder <br>
This folder contains the experiment results for exploring optimum tournamnet size for parent selection as well as mutation probability of an individual gene. It also contains graphs which plot some of the tuning experiments. The outcome was to use a tournament size of 25% of the population (50 individuals) along with a mutation probablity of 0.2. <br>
- the `Data/tuning/tuning_island_model` folder <br>
This folder contains the island_algorithm_tuning which is identical to the normal algorithm however it is looped over several parameters to automate the tuning process. Furthermore it contains the results of all experiments along with some plots. The outcome of these experiments where to use 20 subpopulations with a migration frequency of 5 and a migration magnitude of 4. <br>
- the `Data/final_results` folder <br>
This folder contains the data of all experiments along with the plots comparing the two algorithm on several enemies.
