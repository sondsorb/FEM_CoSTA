import numpy as np


def zero(*args, **kwargs): return 0

def length(x):
    try: 
        return len(x)
    except:
        assert type(x) in [int, float, np.int64, np.float64]
        return 0
