import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, cm
import sys

import sympy as sp


t = sp.symbols('t')
x = sp.symbols('x')
y = sp.symbols('y')
alpha = sp.symbols('alpha')
u = sp.exp(-t/(2-x**2-(y-(1+alpha)/4)**2)) # TODO: calculate a realistic u more properly

#a = np.array([0,1,2]) 
#from sympy.utilities.lambdify import lambdify
#print(u)
#print(u.subs(y,0))
#
#
#f = u.diff(t) - u.diff(x,x) - u.diff(y,y)
#u = lambdify([x,y,t,alpha],f, "numpy")
#print(u(a,a,a,a))
#quit()

## Test: solution 4
#u = 1+sp.sin(2*sp.pi*t+alpha)*sp.cos(2*sp.pi*x)
#f = u.diff(t) - u.diff(x,x) - u.diff(y,y)
#print(f)
# gives corrrect f


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
    global Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, NoM, time_delta
    if mode == 'bugfix':
        Ne = 5
        time_steps = 20
        DNNkwargs = {'n_layers':6,'depth':20, 'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgDNNkwargs = {'n_layers_1':4,'n_layers_2':2,'depth':20,'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}#, 'l1_penalty':0.01}
        LSTMkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
        NoM = 2
        time_delta = 5
    elif mode == 'quick_test':
        Ne = 20
        time_steps = 500
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':8e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':8e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        NoM=3
        time_delta = 0.3 # max 30 steps back
    elif mode == 'full_test':
        Ne = 20
        time_steps = 5000
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20]}
        NoM=10
        time_delta = 0

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
    source = False
print('Using exact source' if source else 'Using unknown source (i.e. guessing zero)')

if len(sys.argv)>3:
    p = int(sys.argv[3])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        #'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs, 
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs, 
        'CoSTA_LSTM':LSTMkwargs, 
        }


f,u = functions.manufacture_solution(u,t,x,y,alpha)
sol = functions.Solution(T=5, f_raw=f, u_raw=u, zero_source=not source, name=f'bp_tst1', time_delta=time_delta)

model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps, a=-1, b=1, NNkwargs=NNkwargs)
extra_tag = '' # for different names when testing specific stuff
figname = None
model_folder = None
model.plot=True
#model.train(figname=figname, model_folder = model_folder)
model.train(figname=figname)
#model.load_weights(model_folder)

figname = None
#figname = None
_ = model.test(interpol = True, figname=figname)
#figname = None
_ = model.test(interpol = False, figname=figname)
