import numpy as np
from numba import jit, prange
from ml_trading_sim.fitness import get_fitness_values
from ml_trading_sim.init_population import init_swarm_with_range

@jit(nopython=True,fastmath = True)
def order_minmax(upper_limit,under_limit):
    upper_limit_old = upper_limit.copy()
    for i in prange(upper_limit.shape[0]):
        if upper_limit[i] < under_limit[i]:
            upper_limit[i] = under_limit[i]
            under_limit[i] = upper_limit_old[i]
            
    return upper_limit,under_limit    


@jit(nopython=True,fastmath = True)
def mutation_7(pop_to_select,mutationrate,upper_limit,under_limit):
    mutateornot = np.random.binomial(1, (mutationrate),(pop_to_select.shape[0]))
    keep_in_memory_orig_data = (1-mutateornot)
    save = np.zeros((pop_to_select.shape[0]))

    for i in prange(pop_to_select.shape[0]):
        if under_limit[i] == upper_limit[i]:
            save[i] = pop_to_select[i]

        else:
            save[i] = np.random.uniform(under_limit[i] ,upper_limit[i])

    save = save*mutateornot
    pop_to_select = pop_to_select*keep_in_memory_orig_data
    pop_to_select = pop_to_select+save

    return pop_to_select

@jit(nopython=True,fastmath = True)
def inner_improvement(old_best_weights,best_weights,best_fitness_memory,min_range,max_range,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count):
    meandistance = ((best_weights-old_best_weights)/2)
    upper_limit = best_weights + meandistance
    under_limit = best_weights - meandistance
    upper_limit = np.where(upper_limit>max_range,max_range,upper_limit)
    under_limit = np.where(under_limit<min_range,min_range,under_limit)
    upper_limit,under_limit = order_minmax(upper_limit,under_limit)
    first_round = False
    meandist_sum = (((np.sum(np.abs(upper_limit))/upper_limit.shape[0])+(np.sum(np.abs(under_limit)))/upper_limit.shape[0]))/2
    mutationrate = 0.7
    No_improvement = 0
    cut_out = 10
    cut_0 = 0
    cut_1 = int(cut_out/4) 
    cut_2 = int(cut_out/2) 
    cut_3 = int(cut_out*(3/4)) 

    for d in prange(100000):
        
        if No_improvement > cut_out:
            return best_weights, best_fitness_memory

        if (No_improvement == cut_0) and (first_round == True): 
            first_round = False
            upper_limit = np.where(upper_limit == 0.0,meandist_sum,0)
            under_limit = np.where(under_limit == 0.0,(-1)*meandist_sum,0)
            upper_limit = best_weights + upper_limit
            under_limit = best_weights - under_limit
            upper_limit,under_limit = order_minmax(upper_limit,under_limit)
            upper_limit = np.where(upper_limit>1.0,1.0,upper_limit)
            under_limit = np.where(under_limit<-1.0,-1.0,under_limit)
                        
        elif No_improvement == cut_1:
            first_round = True
            upper_limit = np.where(upper_limit == 0,meandist_sum,0)
            under_limit = np.where(under_limit == 0,(-1)*meandist_sum,0)
            upper_limit = best_weights + upper_limit
            under_limit = best_weights - under_limit
            upper_limit,under_limit = order_minmax(upper_limit,under_limit)
            upper_limit = np.where(upper_limit>1.0,1.0,upper_limit)
            under_limit = np.where(under_limit<-1.0,-1.0,under_limit)
            
        elif No_improvement == cut_2:
            upper_limit = np.where(upper_limit == 0,(1/10)*meandist_sum,0)
            under_limit = np.where(under_limit == 0,((-1/10))*meandist_sum,0)
            upper_limit = best_weights + upper_limit
            under_limit = best_weights - under_limit
            upper_limit,under_limit = order_minmax(upper_limit,under_limit)
            upper_limit = np.where(upper_limit>1.0,1.0,upper_limit)
            under_limit = np.where(under_limit<-1.0,-1.0,under_limit)
            
        elif No_improvement == cut_3:
            upper_limit = np.where(upper_limit == 0,(1/10)*meandist_sum,0)
            under_limit = np.where(under_limit == 0,(-1/10)*meandist_sum,0)
            upper_limit = best_weights + upper_limit
            under_limit = best_weights - under_limit
            upper_limit,under_limit = order_minmax(upper_limit,under_limit)
            upper_limit = np.where(upper_limit>1.0,1.0,upper_limit)
            under_limit = np.where(under_limit<-1.0,-1.0,under_limit)
            
        new_weights = mutation_7(best_weights,0.7,upper_limit,under_limit)
        new_weights_fit = new_weights.reshape((1, new_weights.shape[0]))        
        pop_fitness,_,_ = get_fitness_values(new_weights_fit,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count)
        pop_fitness = pop_fitness[0]

        if best_fitness_memory < pop_fitness:
            pop_fitness,new_weights = best_model_selection_test(pop_fitness,best_fitness_memory,new_weights,best_weights,under_limit,upper_limit,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count,mutationrate)
            old_best_weights = best_weights.copy()
            best_weights = new_weights.copy()
            best_fitness_memory = pop_fitness
            best_weights,best_fitness_memory = inner_improvement(old_best_weights,best_weights,best_fitness_memory,min_range,max_range,nn_architecture,observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count)
            No_improvement = 0
            
        else:
            No_improvement = No_improvement + 1
        
    return best_weights, best_fitness_memory

@jit(nopython=True,fastmath = True)
def best_model_selection_test(pop_fitness,best_fitness_memory,new_best_weights,best_weights,min_range,max_range,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count,mutationrate):
    units = 10+1
    weights_arr = np.zeros((best_weights.shape[0],units))
    fit_arr = np.zeros(units)

    model_count = 0
    first_pop = pop_fitness
    first_weights = new_best_weights
    for i in prange(units):
        
        new_weights = mutation_7(best_weights,mutationrate,max_range,min_range)
        new_weights_fit = new_weights.reshape((1, new_weights.shape[0]))
        pop_fitness,_,_ = get_fitness_values(new_weights_fit,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count)
        pop_fitness = pop_fitness[0]
        
        if best_fitness_memory < pop_fitness:
            weights_arr[:,model_count] = new_weights
            fit_arr[model_count] = pop_fitness
            model_count += 1
            
    model_count += 1
    filtered_models = np.zeros((new_weights.shape[0],model_count))
    fit_filt = np.zeros((model_count))    
    filtered_models[:,-1] = first_weights
    fit_filt[-1] = first_pop 
    
    for m in prange(model_count-1):
        filtered_models[:,m] = weights_arr[:,m]
        fit_filt[m] = fit_arr[m] 
        
    fit_filt_exponential = fit_filt*fit_filt*fit_filt
    fit_filt_exp_sum = np.sum(fit_filt_exponential)
    avgw = np.zeros(filtered_models.shape[0])
    
    for w in prange(filtered_models.shape[0]):
        for m in prange(filtered_models.shape[1]):
            avgw[w] += (filtered_models[w,m]*fit_filt_exponential[m])
        avgw[w] = avgw[w]/fit_filt_exp_sum
    
    scores = np.zeros(filtered_models.shape[1])

    for m in prange(filtered_models.shape[1]):
        scores[m] = np.sum(np.abs(filtered_models[:,m] - avgw))

    best_weights = filtered_models[:,np.argmin(scores)]
    best_fitness = fit_filt[np.argmin(scores)]

    return best_fitness, best_weights

@jit(nopython=True,fastmath = True)
def random_walker_7(Rounds,init_weights,min_range,max_range,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count,mutationrate=0.7,mutationrate_decrease=0.2,fitness_no_improve_times_limit=15): 
    
    fitness_no_improve_times = 0.0
    fitness_no_improve_times_noreset=1
    small_mut = False
    end_switch = False
    best_fitness_memory= 0.0
    best_weights= init_weights[0].copy()
    new_weights = init_weights[0].copy()
    old_best_weights = np.zeros(init_weights.shape[1])
    old_min_range = min_range.copy()
    old_max_range = max_range.copy()
    
    for z in prange(Rounds):
        
        new_weights_fit = new_weights.reshape((1, new_weights.shape[0]))
        pop_fitness,_,_ = get_fitness_values(new_weights_fit,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count)
        pop_fitness = pop_fitness[0]

        if fitness_no_improve_times == fitness_no_improve_times_limit*(1/5):
            min_range = best_fitness_memory-old_min_range * (1/5)
            max_range = best_fitness_memory+old_max_range * (1/5)
            max_range,min_range = order_minmax(max_range,min_range)
            max_range = np.where(max_range>1.0,1.0,max_range)
            min_range = np.where(min_range<-1.0,-1.0,min_range)
            
        if fitness_no_improve_times == fitness_no_improve_times_limit*(2/5):
            min_range = best_fitness_memory-old_min_range * (1/10)
            max_range = best_fitness_memory+old_max_range * (1/10)
            max_range,min_range = order_minmax(max_range,min_range)
            max_range = np.where(max_range>1.0,1.0,max_range)
            min_range = np.where(min_range<-1.0,-1.0,min_range)
            
        if fitness_no_improve_times == fitness_no_improve_times_limit*(3/5):
            min_range = best_fitness_memory-old_min_range * (1/20)
            max_range = best_fitness_memory+old_max_range * (1/20)
            max_range,min_range = order_minmax(max_range,min_range)
            max_range = np.where(max_range>1.0,1.0,max_range)
            min_range = np.where(min_range<-1.0,-1.0,min_range)
            
        if fitness_no_improve_times == fitness_no_improve_times_limit*(4/5):
            min_range = best_fitness_memory-old_min_range * (1/40)
            max_range = best_fitness_memory+old_max_range * (1/40)
            max_range,min_range = order_minmax(max_range,min_range)
            max_range = np.where(max_range>1.0,1.0,max_range)
            min_range = np.where(min_range<-1.0,-1.0,min_range)
        
        if best_fitness_memory < pop_fitness:
            min_range = old_min_range.copy()
            max_range = old_max_range.copy()
            pop_fitness,new_weights = best_model_selection_test(pop_fitness,best_fitness_memory,new_weights,best_weights,min_range,max_range,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count,mutationrate)
            best_weights = new_weights.copy()
            best_fitness_memory = pop_fitness
            best_weights,best_fitness_memory = inner_improvement(old_best_weights,best_weights,best_fitness_memory,min_range,max_range,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count)
            old_best_weights = best_weights.copy()
            fitness_no_improve_times = 0
            did_better_result = True
                        
        elif pop_fitness<=best_fitness_memory:
            fitness_no_improve_times += 1
            fitness_no_improve_times_noreset += 1 
            
        if (fitness_no_improve_times > fitness_no_improve_times_limit) and (end_switch == False):
            fitness_no_improve_times = 0
            
            if mutationrate > mutationrate_decrease+0.05: 
                mutationrate -= mutationrate_decrease
                
            elif (mutationrate <= mutationrate_decrease+0.05)  and (small_mut == False):
                mutationrate_decrease = 0.05
                small_mut = True
                
            elif (small_mut == True) and (mutationrate > mutationrate_decrease+0.03):
                mutationrate -= mutationrate_decrease
                end_switch = True
                
        if (fitness_no_improve_times > fitness_no_improve_times_limit) and (end_switch == True):
            return best_fitness_memory,best_weights   
            
        new_weights = mutation_7(best_weights,mutationrate,max_range,min_range)

        if np.remainder(fitness_no_improve_times_noreset, 100) == 0:
            #np.save('random_walker_7_walking.npy', best_weights )
            print("¤¤¤¤¤¤¤¤¤ROUND_NRO:",z,"  ¤¤¤¤¤¤¤¤¤")
            print("Best fitness in memory",best_fitness_memory)
            #print("Best fitness of this iteration ", pop_fitness)
            print("Mutationrate",mutationrate)
            print("Fitness haven't improved in",fitness_no_improve_times," iterations")  
        
    return best_fitness_memory,best_weights

@jit(nopython=True, fastmath = True)
def run_optimizer(optimizer_iterations,fitness_no_improve_times_limit,mutationrate_decrease,mutationrate,Number_of_models,pop_size,wandb_arr,feature_encoding_vars,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count):  
    ensemble_weights_arr = np.zeros((Number_of_models,wandb_arr.shape[0]))
    
    for i in prange(Number_of_models):
        print("Model",i)
        population,var_list = init_swarm_with_range(pop_size,wandb_arr,feature_encoding_vars)
        best_fitness_memory,best_weights = random_walker_7(optimizer_iterations,population,var_list[:,0],var_list[:,1],nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count,mutationrate=0.7,mutationrate_decrease=0.2,fitness_no_improve_times_limit=550)
        ensemble_weights_arr[i,:] = best_weights
                
    return ensemble_weights_arr

