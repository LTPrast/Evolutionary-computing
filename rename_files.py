import os

filenames = ['best_solution', 'max_fitness', 'mean_fitness', 'set_up', 'std_fitness']
tsizes = [10,20,30,40,50]
sigmas = [0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
for tsize in tsizes:
    for sigma in sigmas:
        exp = f'TSP_{tsize}_sigma_{sigma}'
        exp_c = f'tsize_{tsize}_sigma_{sigma}'
        
        # Rename folder
        os.rename('Data/tuning/tuning_simple_model_params/'+exp, 'Data/tuning/tuning_simple_model_params/'+exp_c)

        for file in filenames:

            # Rename file
            os.rename(f'./Data/tuning/tuning_simple_model_params/{exp_c}/{exp}_{file}.csv', f'./Data/tuning/tuning_simple_model_params/{exp_c}/{exp_c}_{file}.csv')