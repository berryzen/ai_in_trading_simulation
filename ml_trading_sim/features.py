import numpy as np
from numba import jit, prange, float64
from ml_trading_sim.feature_engineering import minmax_scaler, sma, csmadist, remove_outliers, remove_outliers_price, _ewma

@jit(nopython=True,fastmath = True)
def feature1(featw, observed_data,sim_len):
    ch0back1 = sma(observed_data,int(1000))
    return sma(observed_data,int(3))/(ch0back1)

@jit(nopython=True,fastmath = True)
def feature2(featw, observed_data,sim_len):
    ch0back1 = sma(observed_data,int(100))
    return sma(observed_data,int(3))/(ch0back1)

@jit(nopython=True,fastmath = True)
def feature3(featw, observed_data,sim_len):
    ch0back1 = sma(observed_data,int((10)))
    return sma(observed_data,int(2))/(ch0back1)

@jit(nopython=True,fastmath = True)
def feature4(featw, observed_data,sim_len):
    ch0back4 = _ewma(observed_data, int(np.abs(250)))
    return sma(observed_data,int(np.abs(3)))/((ch0back4))

@jit(nopython=True,fastmath = True)
def feature5(featw, observed_data,sim_len):
    ch0back4 = _ewma(observed_data, int(np.abs(25)))
    return sma(observed_data,int(np.abs(2)))/((ch0back4))

@jit(nopython=True,fastmath = True)
def feature6(featw, observed_data,sim_len):
    ch0back4 = _ewma(observed_data, int(np.abs(3)))
    return observed_data/((ch0back4))
                       
@jit(nopython=True,fastmath = True)
def feature7(featw, observed_data,sim_len):
    pctsave = np.zeros(observed_data.shape[0])
    pctcl0se2 = observed_data[int(np.abs(500)):]/observed_data[:-1*int(500)]
    pctsave[int(np.abs(500)):] = pctcl0se2
    return _ewma(pctsave, int(3))
                   
@jit(nopython=True,fastmath = True)
def feature8(featw, observed_data,sim_len):
    pctsave = np.zeros(observed_data.shape[0])
    pctcl0se2 = observed_data[int(np.abs(50)):]/observed_data[:-1*int(50)]
    pctsave[int(np.abs(50)):] = pctcl0se2
    return _ewma(pctsave, int(2))
                   
@jit(nopython=True,fastmath = True)
def feature9(featw, observed_data,sim_len):
    pctsave = np.zeros(observed_data.shape[0])
    pctcl0se2 = observed_data[int(np.abs(5)):]/observed_data[:-1*int(5)]
    pctsave[int(np.abs(5)):] = pctcl0se2
    return pctsave

@jit(nopython=True,fastmath = True)
def feature10(featw, observed_data,sim_len):
    acc3ler_arr = np.zeros(observed_data.shape[0])
    pctcl0se_acc3_1 = observed_data[int(np.abs(10)):]/observed_data[:-1*int(np.abs(10))]
    pctcl0se_acc3_2 = observed_data[int(np.abs(1000)):]/observed_data[:-1*int(np.abs(1000))]
    acc3ler_arr[int(np.abs(10)):] = pctcl0se_acc3_1
    acc3ler_arr[int(np.abs(1000)):] = acc3ler_arr[int(np.abs(1000)):]/pctcl0se_acc3_2 
    return acc3ler_arr
                   
@jit(nopython=True,fastmath = True)
def feature11(featw, observed_data,sim_len):
    acc3ler_arr = np.zeros(observed_data.shape[0])
    pctcl0se_acc3_1 = observed_data[int(np.abs(3)):]/observed_data[:-1*int(np.abs(3))]
    pctcl0se_acc3_2 = observed_data[int(np.abs(100)):]/observed_data[:-1*int(np.abs(100))]
    acc3ler_arr[int(np.abs(3)):] = pctcl0se_acc3_1
    acc3ler_arr[int(np.abs(100)):] = acc3ler_arr[int(np.abs(100)):]/pctcl0se_acc3_2 
    return acc3ler_arr
                                   
@jit(nopython=True,fastmath = True)
def feature12(featw, observed_data,sim_len):
    acc3ler_arr = np.zeros(observed_data.shape[0])
    pctcl0se_acc3_1 = observed_data[int(np.abs(1)):]/observed_data[:-1*int(np.abs(1))]
    pctcl0se_acc3_2 = observed_data[int(np.abs(10)):]/observed_data[:-1*int(np.abs(10))]
    acc3ler_arr[int(np.abs(1)):] = pctcl0se_acc3_1
    acc3ler_arr[int(np.abs(10)):] = acc3ler_arr[int(np.abs(10)):]/pctcl0se_acc3_2 
    return acc3ler_arr

@jit(nopython=True,fastmath = True)
def feature13(featw, observed_data,sim_len):
    pctsave = np.zeros(observed_data.shape[0])
    pctcl0se2 = observed_data[int(np.abs(1)):]/observed_data[:-1*int(1)]
    pctsave[int(np.abs(1)):] = pctcl0se2
    return pctsave

@jit(nopython=True,fastmath = True)
def feature14(featw, observed_data,sim_len):
    pctsave = np.zeros(observed_data.shape[0])
    pctcl0se2 = observed_data[int(np.abs(2)):]/observed_data[:-1*int(2)]
    pctsave[int(np.abs(2)):] = pctcl0se2
    return _ewma(pctsave, int(7))                   

@jit(nopython=True,fastmath = True)
def features(featw, observed_data,sim_len,selectfeature):
    if selectfeature == 1:
        return feature1(featw, observed_data,sim_len)
    if selectfeature == 2:
        return feature2(featw, observed_data,sim_len)
    if selectfeature == 3:
        return feature3(featw, observed_data,sim_len)
    if selectfeature == 4:
        return feature4(featw, observed_data,sim_len)
    if selectfeature == 5:
        return feature5(featw, observed_data,sim_len)
    if selectfeature == 6:
        return feature6(featw, observed_data,sim_len)
    if selectfeature == 7:
        return feature7(featw, observed_data,sim_len)
    if selectfeature == 8:
        return feature8(featw, observed_data,sim_len)
    if selectfeature == 9:
        return feature9(featw, observed_data,sim_len)
    if selectfeature == 10:
        return feature10(featw, observed_data,sim_len)
    if selectfeature == 11:
        return feature11(featw, observed_data,sim_len)
    if selectfeature == 12:
        return feature12(featw, observed_data,sim_len)
    if selectfeature == 13:
        return feature13(featw, observed_data,sim_len)
    if selectfeature == 14:
        return feature14(featw, observed_data,sim_len)

@jit(nopython=True,fastmath = True)
def features_channels(featw, observed_data,sim_len):
    remove_nans = 1100
    sim_len = sim_len+remove_nans
    observed_data = observed_data[:sim_len]
    channels = np.ones((observed_data.shape[0],14), dtype=float64) 

    for i in prange(14):
        channels[:,i] =  features(featw, observed_data,sim_len,selectfeature = (i+1))
        
    channels = channels[remove_nans:,:]
    return channels

# get min/max values from observed dataset and dataset with features
@jit(nopython=True,fastmath = True)
def feature_optimization_func(featw, observed_data,sim_len):
    
    channels = features_channels(featw, observed_data,sim_len)
    
    for i in prange(channels.shape[1]):
        data,minim,maxim = remove_outliers(channels[:,i])
        channels[:,i] = minmax_scaler(data,minim,maxim)
        
    return channels

@jit(nopython=True,fastmath = True)
def get_minmax(WandB_arr,nn_architecture, observed_data,price_data,sim_len,max1_limit, max2_limit,featvars_count):
    featw = WandB_arr[:featvars_count]
    WandB_arr = WandB_arr[featvars_count:]
    channels,minim,maxim = feature_optimization_func_get_minmax(featw, observed_data,sim_len)
    return minim,maxim

@jit(nopython=True,fastmath = True)
def feature_optimization_func_give_minmax(featw, observed_data,sim_len,min_arr,max_arr):
    channels = features_channels(featw, observed_data,sim_len)
    collect_min = np.zeros(channels.shape[1])
    collect_max = np.zeros(channels.shape[1])

    for i in prange(channels.shape[1]):
        data,minim,maxim = remove_outliers_minmax(channels[:,i],min_arr[i],max_arr[i])
        channels[:,i] = minmax_scaler(data,minim,maxim)
    
    return channels

#Testidataan oma remove outliers näillä minmaxeilla
@jit(nopython=True,fastmath = True)
def feature_optimization_func_get_minmax(featw, observed_data,sim_len):

    channels = features_channels(featw, observed_data,sim_len)

    collect_min = np.zeros(channels.shape[1])
    collect_max = np.zeros(channels.shape[1])
    for i in prange(channels.shape[1]):
        data,minim,maxim = remove_outliers(channels[:,i])
        collect_min[i] = minim
        collect_max[i] = maxim
        channels[:,i] = minmax_scaler(data,minim,maxim)
    

    return channels,collect_min,collect_max

@jit(nopython=True,fastmath = True)
def remove_outliers_minmax(data,mini,maxi): 
    minim = mini
    maxim = maxi
    data = np.where(data < maxim, data, maxim)
    data = np.where(data > minim, data, minim)

    return data,minim,maxim



