import numpy as np
from numba import jit, prange
from numba import float64    

#Simple moving avarage
@jit(nopython=True,fastmath = True)
def sma(datasma,window): 
    smadata = np.ones(datasma.shape[0])
    window = int(window)
    for i in prange(datasma.shape[0]-window):
        smadata[i+window] = np.sum(datasma[i:i+window])/window
    return smadata

#Relative distance between points
@jit(nopython=True,fastmath = True)
def csmadist(data_mid,data_sema): 
    semadata = np.ones(data_mid.shape[0])
    nonzero_divier = 0.00001
    data_sema += nonzero_divier
    semadata = data_mid / data_sema
    return semadata

#reduces overfiting
@jit(nopython=True,fastmath = True)
def remove_outliers(data): 
    minim = np.quantile(data,0.01)
    maxim = np.quantile(data,0.99)
    data = np.where(data < maxim, data, maxim)
    data = np.where(data > minim, data, minim)

    return data,minim,maxim

#reduces overfiting
@jit(nopython=True)
def remove_outliers_price(data): 
    minim = np.quantile(data,0.05)
    maxim = np.quantile(data,0.95)
    data = np.where(data < maxim, data, maxim)
    data = np.where(data > minim, data, minim)

    return data,minim,maxim


 #Scales feature data between 0 and 1
@jit(nopython=True,fastmath = True)
def minmax_scaler(data,minim,maxim):
    datain = data
    datain -= minim
    datain /= maxim - minim
    return datain

@jit(nopython=True,fastmath = True)
def _ewma(arr_in, window):
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma