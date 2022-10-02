# Studying the Effects of Preserving Diversity in an Evolutionary Algorithm on Performance and Robustness

This repository was built to explore the effect of superimposing a spatial component onto a basic Evolutionary Algorithm to analyze how this effects the preformance and robustness of an algorithm. The spatial component got introduced by splitting the population into subpopulations (islands) between which individuals migrate at agiven frequency, the idea is that this allows different populations to converge to different solutions, thus maintaining diversity and exploring a larger section of the solution space. The central points of interests were the maximum and mean fitness values and how these progress through generations as well as how conistently an algorithm converges to good solutions. An evolutionary algorithm was constructed to optimize the weight of nodes of a neural network with the aim of creating a specialist agent in the evoman framework tested on three of the eight potential enemies. 

## Process
The approach to this project was to construct both the basic algorithm as well as the island model algorithms. Both algorithms have the same core to allow a fair comparisson. Namley parent selection is preformed using a tournament selection, offspring are created using two methods, a simple mutation as well as simple arithmetic combination followed by a mutation and the population size is kept constant by getting rid of the indiviudlas with the worst fitness values. 

Following this both algorithms were tuned to allow a fair comparisson. First an attempt was made to identify the best form of mutating indiviudlas i.e.  from which distribution and with which standrad deviations samples should be drawn to mutate genes in an indiviudal genome. Secondly for the basic algorithm two parameters were tuned namely the tournament size for parent selction as well as the mutation proability, the probability of mutating an individual gene during the mutation process. Given these values the island model was tuned for three further parameters: the number of subpopulations (islands), the migration frequency (how often individuals migrate) and the migration magnitude (how many individuals move betwen islands).   

## Repository Set-Up
This section will give a brief overview of the differen files and folders in the repository.

### Central Files
Main algorithm files:
`controlller.py` is the neural network. <br>
`basic_algorithm.py` contains the basic evolutionary algorithm. <br>
`island_algorithm.py` contains the central algorithm for the island model. <br>

Methods files: <br>
`parent_selection_methods.py` contains functions for the selction of individuals to produce offspring. <br>
`survival_methods.py` contains the functions to maintain population size and hence delte a certain number of individuals. <br>
`mutation_recombination_methods.py` contains the functions by which offspring is created given the genome of the parents. <br>
`island_functions.py` contains the functions for the island model responsible for creating offspring and migration. <br>

### Analysis files
The results of the experiments are stored in csv files, the following files were used for analysis of these results: <br>
`plotting_functions.py` contains functions to plot results in different ways depending on the application. <br>
`create_plots.py` calls upon these functions to create plots. <br>
`mean_max_csv.py` proviudes csv files with the final mean and max fitness values of several experiments. <br>
`run_best_ind_gain.py` ........................................................................... <br>

### evoman folder
This folder contains the evoman framework which should not be altered.

### Data folder
This folder contains all our data folder and includes our tuning and final run folders.

### trial_island_model_10_runs folder
This folder contains the results of a trial run with the island model along with graphs plotting the devlopment of the different islands to develop an intution for the working of the model. 

### Data/tuning/tuning_mutation_distribution folder
This folder contains the results of experimentation with different distribution types (mainly unifrom and gaussian) from which mutations to individual nodes are drawn. Furthermore in contains graphs of the outcomes. The result of these experiments was to use a gaussian distribution with mean and a standrad deviation of 0.1.

### Data/tuning/tuning_simple_model_params folder
This folder contains the experiment results for exploring optimum tournamnet size for parent selection as well as muatation probability of an individual gene. It also contains graphs which plot some of the signifcant experiments. The outcome was to use a tournament size of 25% of the population along with a mutation probablity of 0.2.

### Data/tuning/tuning_island_model folder
This folder contains the island_algorithm_tuning which is identical to the normal algorithm however it is looped over several paramters to optimize the tuning process. Furthermore it contains the results of all experiments along with some plots. The outcome of these experiments where to use XXX subpopulations with a migration frequency of XXX and a migration magnitude of XXX.

### Data/final_results folder
This folder contains the data of all experiments along with the plots comparing the two algorithm on several enemies.
