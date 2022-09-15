################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np

sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

experiment_name = 'dummy_demo'
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


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

#up and down limit for weights neurons
dom_u = 1
dom_l = -1

npop = 60 #population per gen
ngen = 20 #numeber of gen gen

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def new_population():
    "adds new population except for the best one"
    new_pop = np.random.uniform(dom_l, dom_u, (npop-1, n_vars))
    return new_pop

def new_quarter_population(pop, fit_pop):
    "removes the worst quarter of the population and adds new random inplace"
    half_pop = int(npop/4) # quarter of the population
    order = np.argsort(fit_pop)
    best = order[half_pop:]
    new_pop = np.random.uniform(dom_l, dom_u, (half_pop, n_vars))
    best_pop = pop[best]
    pop = np.vstack((new_pop, best_pop))
    return pop

    

print( '\nNEW EVOLUTION\n')

pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

for i in range(ngen):
    print('generation', i)
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    print('best:', best, 'mean:', mean, 'std:', std, '\n')
    
    #best_pop = pop[best]
    #new_pop = new_population()

    pop = new_quarter_population(pop, fit_pop)
    



