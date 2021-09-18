import numpy as np
import quadrature
import FEM
import solvers
from matplotlib import pyplot as plt

NNkwargss=[
        {'l':6,'n':80},
        #{'l':2, 'n':16},
        #{'l':3, 'n':16},
        #{'l':4, 'n':16},
        #{'l':2, 'n':12},
        #{'l':3, 'n':12},
        #{'l':4, 'n':12},
        ]

for NNkwargs in NNkwargss:
    femscores = []
    costascores = []
    for sol in range(3,4):
        #print(f'\nsol: {sol}')
        model = solvers.Costa(p=1,sol=sol, Ne=10, time_steps=20, T=5, **NNkwargs)
        #model.plot=False
        model.train()
        fs, cs = model.test()
        femscores.append([fs[k] for k in fs])
        costascores.append([cs[k] for k in cs])
    print(femscores)
    print(costascores)
    femscores = np.array(femscores)
    costascores = np.array(costascores)
    print(femscores)
    print(costascores)
    relscore = np.divide(costascores, femscores) # elementwise division
    score = np.mean(relscore)
    print(relscore)
    print('----', score)
