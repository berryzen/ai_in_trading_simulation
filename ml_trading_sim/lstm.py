import numpy as np
from numba import jit, prange

# lstm.py includes recurrent neural network named lstm - long short term memory

def init_lstm_layers(lstm_architecture):
    input_size  = lstm_architecture[0] 
    hidden_dim  = lstm_architecture[1] 
    output_size = lstm_architecture[2]    
    weights = np.ones((hidden_dim*input_size*4)+hidden_dim*4+hidden_dim*hidden_dim*4+hidden_dim*4+hidden_dim*output_size+output_size, dtype=np.float64)

    return weights

@jit(nopython=True,fastmath = True)
def wandb_to_lstm_matrices(wandb_arr,lstm_architecture):
    
    input_size  = lstm_architecture[0] 
    hidden_dim  = lstm_architecture[1] 
    output_size = lstm_architecture[2]
    
    start=0
    end = hidden_dim*input_size
    
    weights_xi = wandb_arr[start:end].reshape((hidden_dim,input_size))
    start=end
    end = end+hidden_dim*input_size
    
    weights_xf = wandb_arr[start:end].reshape((hidden_dim,input_size))
    start=end
    end = end+hidden_dim*input_size
    
    weights_xl = wandb_arr[start:end].reshape((hidden_dim,input_size))
    start=end
    end = end+hidden_dim*input_size
    
    weights_xo =  wandb_arr[start:end].reshape((hidden_dim,input_size))
    
    start=end
    end = end+hidden_dim
    bias_xi = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim
    bias_xf = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim
    bias_xl = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim
    bias_xo = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim*hidden_dim
    weights_hi = wandb_arr[start:end].reshape((hidden_dim,hidden_dim))
    
    start=end
    end = end+hidden_dim*hidden_dim
    weights_hf = wandb_arr[start:end].reshape((hidden_dim,hidden_dim))
    
    start=end
    end = end+hidden_dim*hidden_dim
    weights_hl = wandb_arr[start:end].reshape((hidden_dim,hidden_dim))
    
    start=end
    end = end+hidden_dim*hidden_dim
    weights_ho = wandb_arr[start:end].reshape((hidden_dim,hidden_dim))

    start=end
    end = end+hidden_dim
    bias_hi = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim
    bias_hf = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim
    bias_hl = wandb_arr[start:end].reshape((hidden_dim))
    
    start=end
    end = end+hidden_dim
    bias_ho = wandb_arr[start:end].reshape((hidden_dim))

    start=end
    end = end+hidden_dim*output_size
    fc_weight = wandb_arr[start:end].reshape((hidden_dim,output_size))
    
    start=end
    end = end+output_size
    fc_bias = wandb_arr[start:end].reshape((output_size))
    
    return fc_weight,fc_bias,bias_ho,bias_hl,bias_hf,bias_hi,weights_ho,weights_hl,weights_hf,weights_hi,bias_xo,bias_xl,bias_xf,bias_xi,weights_xi,weights_xf,weights_xl,weights_xo

# main lstm feed forward function
@jit(nopython=True,fastmath = True)
def lstm_feed_forward(next_state,h,c,fc_weight,fc_bias,bias_ho,bias_hl,bias_hf,bias_hi,weights_ho,weights_hl,weights_hf,weights_hi,bias_xo,bias_xl,bias_xf,bias_xi,weights_xi,weights_xf,weights_xl,weights_xo):
    f = forget_gate(next_state, h, weights_hf, bias_hf, weights_xf, bias_xf, c)
    i = input_gate(next_state, h, weights_hi, bias_hi, weights_xi, bias_xi, weights_hl, bias_hl, weights_xl, bias_xl)
    c = cell_state(f,i)
    h = output_gate(next_state, h, weights_ho, bias_ho, weights_xo, bias_xo, c)
  
    return model_output(h, fc_weight, fc_bias),c,h

@jit(nopython=True,fastmath = True)
def forget_gate(x, h, weights_hf, bias_hf, weights_xf, bias_xf, prev_cell_state):
    forget_hidden  = np.dot(weights_hf, h) + bias_hf
    forget_next_state  = np.dot(weights_xf, x) + bias_xf
    return np.multiply( sigmoid(forget_hidden + forget_next_state), prev_cell_state )

@jit(nopython=True,fastmath = True)
def input_gate(x, h, weights_hi, bias_hi, weights_xi, bias_xi, weights_hl, bias_hl, weights_xl, bias_xl):
    ignore_hidden  = np.dot(weights_hi, h) + bias_hi
    ignore_next_state  = np.dot(weights_xi, x) + bias_xi
    learn_hidden   = np.dot(weights_hl, h) + bias_hl
    learn_next_state   = np.dot(weights_xl, x) + bias_xl
    return np.multiply( sigmoid(ignore_next_state + ignore_hidden), np.tanh(learn_next_state + learn_hidden) )

@jit(nopython=True,fastmath = True)
def cell_state(forget_gate_output, input_gate_output):
    return forget_gate_output + input_gate_output

@jit(nopython=True,fastmath = True)
def output_gate(x, h, weights_ho, bias_ho, weights_xo, bias_xo, cell_state):
    out_hidden = np.dot(weights_ho, h) + bias_ho
    out_next_state = np.dot(weights_xo, x) + bias_xo
    return np.multiply( sigmoid(out_next_state + out_hidden), np.tanh(cell_state) )

@jit(nopython=True,fastmath = True)
def model_output(lstm_output, fc_weight, fc_bias):
  return softmax(np.dot( lstm_output,fc_weight) + fc_bias)

#Activation Functions
@jit(nopython=True,fastmath = True)
def sigmoid(X):
    return 1/(1+np.exp(-X))

def tanh_activation(X):
    return np.tanh(X)

@jit(nopython=True,fastmath = True)
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X)
    exp_X = exp_X/exp_X_sum
    return exp_X

