import numpy as np

def mse(x,y):
    return np.mean((np.array(x)-np.array(y))**2)
