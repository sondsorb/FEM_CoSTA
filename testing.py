import numpy as np
import quadrature
import FEM
import solvers
from matplotlib import pyplot as plt

NNkwargss=[
        {'l':6,'n':80, 'lr':1e-5, 'patience':20},
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
        print(f'\nsol: {sol}')
        Ne = 20#50 if sol==3 else 20
        model = solvers.Costa(p=1,sol=sol, Ne=Ne, time_steps=200, T=5, **NNkwargs)
        model.plot=True
        model.train()
        fs, cs = model.test()
        femscores.append([fs[k] for k in fs])
        costascores.append([cs[k] for k in cs])
        #fs, cs = model.test(False)
        #femscores.append([fs[k] for k in fs])
        #costascores.append([cs[k] for k in cs])
    femscores = np.array(femscores)
    costascores = np.array(costascores)
    #print(femscores)
    #print(costascores)
    relscore = np.divide(costascores, femscores) # elementwise division
    score = np.mean(relscore)
    print(relscore)
    print('----', score)
