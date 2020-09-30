import numpy as np
from sklearn.linear_model import LinearRegression



def get_residual(log_ret):
    '''
    @log_ret (dataframe): log return
    Output(numpy array): Residual 
    
    '''
    ret_mean = log_ret.mean(axis = 1)
    log_ret, ret_mean = np.array(log_ret), np.array(ret_mean).reshape(-1,1)
    lm = LinearRegression().fit(ret_mean, log_ret)
    residual = np.array(log_ret) - lm.predict(np.array(ret_mean).reshape(-1,1))
    return residual


def get_e(residual):
    '''
    @ residual (numpy array): residual get from function get_residual: 
    @ Output (1* time) : e square
    '''
    #residual = get_residual(log_ret)
    ## sqrt(h_t)
    std_inv = (1.0/ np.var(residual, axis = 1) ** 0.5).reshape(-1,1)
    
    e = (residual - np.mean(residual, axis = 1).reshape(-1,1)) * std_inv 
    
  
    return e ** 2




