import numpy as np
from numba import jit

# This initializes weights of neuralnetwork for population of ai agents, to get ready to run in simulation 
@jit(nopython=True)
def init_swarm_with_range(pop_size,wandb_arr,feature_encoding_vars):
    var_list = []
    
    for n in range(feature_encoding_vars):
        var_list.append((0.0,1.0))
            
    for n in range(wandb_arr.shape[0]):
        var_list.append((-1.0,1.0))

    var_list = np.asarray(var_list)
    population = init_population(var_list,population_size=pop_size)
    return population,var_list

@jit(nopython=True)
def init_population(var_list,population_size=150):
    population_size = population_size
    population = np.ones((population_size,var_list.shape[0]))
    for i in range(population_size):
        population[i] = init_unit(var_list)
    return population

@jit(nopython=True)
def init_unit(var_list):
    init_variables = np.ones(var_list.shape[0])
    for i in range(init_variables.shape[0]):
        init_variables[i] = np.random.uniform(var_list[i][0],var_list[i][1])
    return init_variables


