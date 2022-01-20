import numpy as np


def zero(*args, **kwargs): return 0

def length(x):
    try: 
        return len(x)
    except:
        assert type(x) in [int, float, np.int64, np.float64]
        return 0

def merge_first_dims(data):
    '''
    Reshapes numpy array from shape (a,b,...) to (a*b,...), such that input[i,j,...]==output[j*input.shape[0]+i,...]'''
    return np.reshape(
            a = data,
            newshape = (data.shape[0]*data.shape[1], *data.shape[2:]),
            order = 'F')
