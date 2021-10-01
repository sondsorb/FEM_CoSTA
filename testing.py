import numpy as np
import quadrature
import FEM
import solvers
from matplotlib import pyplot as plt
import sys

mode = 'bugfix'
if len(sys.argv)>1:
    mode = {'0':'bugfix',
            '1':'quick_test',
            '2':'full_test'}[sys.argv[1]]
else:
    print('No mode specified')
    print('add arg for mode:')
    print('0: bufix (default)')
    print('1: quick_test')
    print('2: full_test')
    print('syntax e.g.: python testing.py 2')




def set_args(mode=mode):
    print(f'\nTesting with mode "{mode}"...')
    global Ne, time_steps, NNkwargs
    if mode == 'bugfix':
        Ne = 5
        time_steps = 20
        NNkwargs = {'l':4,'n':20,'lr':5e-3,'patience':10}
        NoM = 2
    elif mode == 'quick_test':
        Ne = 20
        time_steps = 500
        NNkwargs = {'l':6,'n':80, 'lr':8e-5, 'patience':20}
        NoM=3
    elif mode == 'full_test':
        Ne = 20
        time_steps = 5000
        NNkwargs = {'l':6,'n':80, 'lr':1e-5, 'patience':20}
        NoM=4

set_args()

if len(sys.argv)>2:
    if sys.argv[2]=='f':
        source = True 
    elif sys.argv[2]=='0':
        source = False
    else:
        print('failed reading source term')
        source = True
else:
    source = True
print('Using exact source' if source else 'Using unknown source (i.e. guessing zero)')

femscores = []
costascores = []
pnnscores = []
for sol in range(1,5):
    print(f'sol: {sol}\n')
    model = solvers.Solvers(p=1,sol=sol, unknown_source = not source, Ne=Ne, time_steps=time_steps, T=5, NoM=NoM, **NNkwargs)
    #model.plot=False
    model.train()
    model.plot=True
    #fs, cs = model.test()
    #femscores.append([fs[k] for k in fs])
    #costascores.append([cs[k] for k in cs])
    fs, cs, ps = model.test(False)
    femscores.append([fs[k] for k in fs])
    costascores.append([cs[k] for k in cs])
    pnnscores.append([ps[k] for k in cs])
#femscores = np.array(femscores)
#costascores = np.array(costascores)
#pnnscores = np.array(costascores)
##print(femscores)
##print(costascores)
#relscore = np.divide(costascores, femscores) # elementwise division
#score = np.mean(relscore)
#print(relscore)
#print('----', score)
