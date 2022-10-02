# Studying the Effects of Preserving Diversity in an Evolutionary Algorithm on Performance and Robustness

This repository was built to explore the effect of superimposing a spatial component onto a basic Evolutionary Algorithm to analyze how this effects the preformance and robustness of an algorithm. The spatial component got introduced by splitting the population into subpopulations (islands) between which individuals migrate at agiven frequency, the idea is that this allows different populations to converge to different solutions, thus maintaining diversity and exploring a larger section of the solution space. The central points of interests were the maximum and mean fitness values and how these progress through generations as well as how conistently an algorithm converges to good solutions. An evolutionary algorithm was constructed to optimize the weight of nodes of a neural network with the aim of creating a specialist agent in the evoman framework tested on three of the eight potential enemies. 

## Process
The approach to this project was to construct both the basic algorithm as well as the island model algorithms. Both algorithms have the same core to allow a fair comparisson. Namley parent selection is preformed using a tournament selection, offspring are created using two methods, a simple mutation as well as simple arithmetic combination followed by a mutation and the population size is kept constant by getting rid of the indiviudlas with the worst fitness values. 

Following this both algorithms were tuned to allow a fair comparisson. First an attempt was made to identify the best form of mutating indiviudlas i.e.  from which distribution and with which standrad deviations samples should be drawn to mutate genes in an indiviudal genome. Secondly for the basic algorithm two parameters were tuned namely the tournament size for parent selction as well as the mutation proability, the probability of mutating an individual gene during the mutation process. Given these values the island model was tuned for three further parameters: the number of subpopulations (islands), the migration frequency (how often individuals migrate) and the migration magnitude (how many individuals move betwen islands).   

controlller.py is the neural network <br>
Evoman file is the framework of the game which you should not change <br>
dummy_demo file is the repository that keeps all the information of your runs <br>
individual_demo file is the repository that keeps all the information of the the run of the demo that they provided <br>
