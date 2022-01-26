import numpy as np
from numba import jit, prange
from ml_trading_sim.lstm import wandb_to_lstm_matrices, lstm_feed_forward
from ml_trading_sim.environment import RelativeTrade

# Fitness.py computes cumulative product of profit(fitness) from all different agents in their simulation runs. Then this result is returned for optimizer

# Single tradingbot is running through single simulation run to compute fitness
@jit(nopython=True, fastmath = True)
def personalfitness(wandb_arr,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count):

    fc_weight,fc_bias,bias_ho,bias_hl,bias_hf,bias_hi,weights_ho,weights_hl,weights_hf,weights_hi,bias_xo,bias_xl,bias_xf,bias_xi,weights_xi,weights_xf,weights_xl,weights_xo = wandb_to_lstm_matrices(wandb_arr,nn_architecture)
    h = np.zeros(nn_architecture[1], dtype=np.float64)
    c = np.zeros(nn_architecture[1], dtype=np.float64)    
    featw = wandb_arr[:featvars_count]
    wandb_arr = wandb_arr[featvars_count:]

    if observed_data.ndim == 1:
        channel_data = feature_optimization_func(featw, observed_data,sim_len).astype(np.float64)
    else: 
        channel_data = observed_data.astype(np.float64)

    price_data = price_data[:,1100:]
    env = RelativeTrade(channel_data,price_data,sim_len,max1_limit, max2_limit)
    reward,maxreward,maxwallet2,_,next_state = env.reset()
    action,c,h = lstm_feed_forward(next_state.astype(np.float64),h,c,fc_weight,fc_bias,bias_ho,bias_hl,bias_hf,bias_hi,weights_ho,weights_hl,weights_hf,weights_hi,bias_xo,bias_xl,bias_xf,bias_xi,weights_xi,weights_xf,weights_xl,weights_xo)

    for i in prange(channel_data.shape[0]-10):
        reward,maxreward,maxwallet2, done,next_state,_episode_ended = env.step(action)
        if done == True: 
            reward = 0.0
            return reward,maxreward,maxwallet2            
        if _episode_ended == True:
            return reward,maxreward,maxwallet2
            
        action,c,h = lstm_feed_forward(next_state.astype(np.float64),h,c,fc_weight,fc_bias,bias_ho,bias_hl,bias_hf,bias_hi,weights_ho,weights_hl,weights_hf,weights_hi,bias_xo,bias_xl,bias_xf,bias_xi,weights_xi,weights_xf,weights_xl,weights_xo)
            
    return reward,maxreward,maxwallet2

# Population of multiple tradingbots is running through simulation, result is returned to optimizer
@jit(nopython=True,fastmath = True)
def get_fitness_values(population,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count):
    fitness_values = np.ones((population.shape[0]))
    maxfitness_values = np.ones((population.shape[0]))
    maxfitness2_values = np.ones((population.shape[0]))
    for i in prange(fitness_values.shape[0]):
        fitness_values[i],maxfitness_values[i],maxfitness2_values[i] = personalfitness(population[i,:],nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count)
    return fitness_values,maxfitness_values,maxfitness2_values

# In testing this is used for best performing agent to run single run, first running through training data and then continuing to testing data.
@jit(nopython=True,fastmath = True)
def personalfitness_test(ensemble_weights_arr,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count,min_arr,max_arr,trainig_len):
    nro_of_models = ensemble_weights_arr.shape[0]
    
    fc_weight,fc_bias,bias_ho,bias_hl,bias_hf,bias_hi,weights_ho,weights_hl,weights_hf,weights_hi,bias_xo,bias_xl,bias_xf,bias_xi,weights_xi,weights_xf,weights_xl,weights_xo = wandb_to_lstm_matrices(ensemble_weights_arr[0,:],nn_architecture)
    
    fc_weight_arr = np.zeros((nro_of_models,fc_weight.shape[0],fc_weight.shape[1]), dtype=np.float64)
    fc_bias_arr = np.zeros((nro_of_models,fc_bias.shape[0]), dtype=np.float64)
    bias_ho_arr = np.zeros((nro_of_models,bias_ho.shape[0]), dtype=np.float64)
    bias_hl_arr = np.zeros((nro_of_models,bias_hl.shape[0]), dtype=np.float64)
    bias_hf_arr = np.zeros((nro_of_models,bias_hf.shape[0]), dtype=np.float64)
    bias_hi_arr = np.zeros((nro_of_models,bias_hi.shape[0]), dtype=np.float64)
    weights_ho_arr = np.zeros((nro_of_models,weights_ho.shape[0],weights_ho.shape[1]), dtype=np.float64)
    weights_hl_arr = np.zeros((nro_of_models,weights_hl.shape[0],weights_hl.shape[1]), dtype=np.float64)
    weights_hf_arr = np.zeros((nro_of_models,weights_hf.shape[0],weights_hf.shape[1]), dtype=np.float64)
    weights_hi_arr = np.zeros((nro_of_models,weights_hi.shape[0],weights_hi.shape[1]), dtype=np.float64)
    bias_xo_arr = np.zeros((nro_of_models,bias_xo.shape[0]), dtype=np.float64)
    bias_xl_arr = np.zeros((nro_of_models,bias_xl.shape[0]), dtype=np.float64)
    bias_xf_arr = np.zeros((nro_of_models,bias_xf.shape[0]), dtype=np.float64)
    bias_xi_arr = np.zeros((nro_of_models,bias_xi.shape[0]), dtype=np.float64)
    weights_xi_arr = np.zeros((nro_of_models,weights_xi.shape[0],weights_xi.shape[1]), dtype=np.float64)
    weights_xf_arr = np.zeros((nro_of_models,weights_xf.shape[0],weights_xf.shape[1]), dtype=np.float64)
    weights_xl_arr = np.zeros((nro_of_models,weights_xl.shape[0],weights_xl.shape[1]), dtype=np.float64)
    weights_xo_arr = np.zeros((nro_of_models,weights_xo.shape[0],weights_xo.shape[1]), dtype=np.float64)
   
    fc_weight_arr[0],fc_bias_arr[0],bias_ho_arr[0],bias_hl_arr[0],bias_hf_arr[0],bias_hi_arr[0],weights_ho_arr[0],weights_hl_arr[0],weights_hf_arr[0],weights_hi_arr[0],bias_xo_arr[0],bias_xl_arr[0],bias_xf_arr[0],bias_xi_arr[0],weights_xi_arr[0],weights_xf_arr[0],weights_xl_arr[0],weights_xo_arr[0] = wandb_to_lstm_matrices(ensemble_weights_arr[0,:],nn_architecture)
    for z in prange(ensemble_weights_arr.shape[0]-1):
        fc_weight_arr[z],fc_bias_arr[z],bias_ho_arr[z],bias_hl_arr[z],bias_hf_arr[z],bias_hi_arr[z],weights_ho_arr[z],weights_hl_arr[z],weights_hf_arr[z],weights_hi_arr[z],bias_xo_arr[z],bias_xl_arr[z],bias_xf_arr[z],bias_xi_arr[z],weights_xi_arr[z],weights_xf_arr[z],weights_xl_arr[z],weights_xo_arr[z] = wandb_to_lstm_matrices(ensemble_weights_arr[z+1,:],nn_architecture)

    h_arr = np.zeros((nro_of_models,nn_architecture[1]), dtype=np.float64)
    c_arr = np.zeros((nro_of_models,nn_architecture[1]), dtype=np.float64)
    action_arr = np.zeros((nro_of_models,nn_architecture[2]), dtype=np.float64)

    channel_data = observed_data.astype(np.float64)

    price_data = price_data[:,1100:]
    cum_stack = price_data[0].copy()

    env = RelativeTrade(channel_data,price_data,trainig_len,max1_limit, max2_limit)
    reward,maxreward,maxwallet2,_,next_state = env.reset()
    
    for m in prange(nro_of_models):
        action_arr[m],c_arr[m],h_arr[m] = lstm_feed_forward(next_state.astype(np.float64),h_arr[m],c_arr[m],fc_weight_arr[m],fc_bias_arr[m],bias_ho_arr[m],bias_hl_arr[m],bias_hf_arr[m],bias_hi_arr[m],weights_ho_arr[m],weights_hl_arr[m],weights_hf_arr[m],weights_hi_arr[m],bias_xo_arr[m],bias_xl_arr[m],bias_xf_arr[m],bias_xi_arr[m],weights_xi_arr[m],weights_xf_arr[m],weights_xl_arr[m],weights_xo_arr[m])#full_forward_propagation(next_state, env.get_countofactions, wandb_arr, nn_architecture)
    
    current_action_in = np.zeros(nn_architecture[2])
    for a in prange(nn_architecture[2]):
        current_action_in[a] = np.sum(action_arr[:,a])/nro_of_models
        
    for i in prange(trainig_len):
        reward,maxreward,maxwallet2, done,next_state,_episode_ended = env.step(current_action_in)  
        if np.remainder(i, 1000) == 0:
            print("¤¤¤¤¤¤¤¤¤ROUND_NRO:",i,"  ¤¤¤¤¤¤¤¤¤")
            print("reward",maxwallet2)
                
        for m in prange(nro_of_models):
            action_arr[m],c_arr[m],h_arr[m] = lstm_feed_forward(next_state.astype(np.float64),h_arr[m],c_arr[m],fc_weight_arr[m],fc_bias_arr[m],bias_ho_arr[m],bias_hl_arr[m],bias_hf_arr[m],bias_hi_arr[m],weights_ho_arr[m],weights_hl_arr[m],weights_hf_arr[m],weights_hi_arr[m],bias_xo_arr[m],bias_xl_arr[m],bias_xf_arr[m],bias_xi_arr[m],weights_xi_arr[m],weights_xf_arr[m],weights_xl_arr[m],weights_xo_arr[m])#full_forward_propagation(next_state, env.get_countofactions, wandb_arr, nn_architecture)

        current_action_in = np.zeros(nn_architecture[2])
        for a in prange(nn_architecture[2]):
            current_action_in[a] = np.sum(action_arr[:,a])/nro_of_models
            
    testing_len = sim_len-trainig_len
    env = RelativeTrade(channel_data[trainig_len:,:],price_data[:,trainig_len:],testing_len,max1_limit, max2_limit)
    reward,maxreward,maxwallet2,_,next_state = env.reset()
    
    for m in prange(nro_of_models):
        action_arr[m],c_arr[m],h_arr[m] = lstm_feed_forward(next_state.astype(np.float64),h_arr[m],c_arr[m],fc_weight_arr[m],fc_bias_arr[m],bias_ho_arr[m],bias_hl_arr[m],bias_hf_arr[m],bias_hi_arr[m],weights_ho_arr[m],weights_hl_arr[m],weights_hf_arr[m],weights_hi_arr[m],bias_xo_arr[m],bias_xl_arr[m],bias_xf_arr[m],bias_xi_arr[m],weights_xi_arr[m],weights_xf_arr[m],weights_xl_arr[m],weights_xo_arr[m])#full_forward_propagation(next_state, env.get_countofactions, wandb_arr, nn_architecture)
    
    current_action_in = np.zeros(nn_architecture[2])
    for a in prange(nn_architecture[2]):
        current_action_in[a] = np.sum(action_arr[:,a])/nro_of_models
        
    for i in prange(testing_len):
        reward,maxreward,maxwallet2, done,next_state,_episode_ended = env.step(current_action_in)  
        if np.remainder(i, 1000) == 0:
            print("¤¤¤¤¤¤¤¤¤ROUND_NRO:",i,"  ¤¤¤¤¤¤¤¤¤")
            print("reward",maxwallet2)

        cum_stack[i] = maxwallet2
    
        for m in prange(nro_of_models):
            action_arr[m],c_arr[m],h_arr[m] = lstm_feed_forward(next_state.astype(np.float64),h_arr[m],c_arr[m],fc_weight_arr[m],fc_bias_arr[m],bias_ho_arr[m],bias_hl_arr[m],bias_hf_arr[m],bias_hi_arr[m],weights_ho_arr[m],weights_hl_arr[m],weights_hf_arr[m],weights_hi_arr[m],bias_xo_arr[m],bias_xl_arr[m],bias_xf_arr[m],bias_xi_arr[m],weights_xi_arr[m],weights_xf_arr[m],weights_xl_arr[m],weights_xo_arr[m])#full_forward_propagation(next_state, env.get_countofactions, wandb_arr, nn_architecture)

        current_action_in = np.zeros(nn_architecture[2])
        for a in prange(nn_architecture[2]):
            current_action_in[a] = np.sum(action_arr[:,a])/nro_of_models

    return cum_stack