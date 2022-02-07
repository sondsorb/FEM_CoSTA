import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, cm
import sys
import sympy as sp

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
        Ne = 4
        time_steps = 20
        DNNkwargs = {'n_layers':6,'depth':20, 'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgDNNkwargs = {'n_layers_1':4,'n_layers_2':2,'depth':20,'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}#, 'l1_penalty':0.01}
        LSTMkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
        NoM = 3
        time_delta = 5
    elif mode == 'quick_test':
        Ne = 8
        time_steps = 500
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':8e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':8e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        NoM=4
        time_delta = 0.3 # max 30 steps back
    elif mode == 'full_test':
        Ne = 12
        time_steps = 5000
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20]}
        NoM=10
        time_delta = 0

set_args()

if len(sys.argv)>2:
    if sys.argv[2]=='bp':
        source = 'reduced_source'
    elif sys.argv[2]=='f':
        source = 'exact_source'
    elif sys.argv[2]=='0':
        source = 'zero_source'
    else:
        print('failed reading source term')
        source = 'exact_source'
else:
    source = 'exact_source'
print('Using', source)

if len(sys.argv)>3:
    p = int(sys.argv[3])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        'pgDNN' : NoM,
        'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        'CoSTA_pgDNN' : NoM,
        'CoSTA_LSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs,
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs,
        'CoSTA_LSTM':LSTMkwargs,
        }

xa,xb,ya,yb = -1,1,-1,1
for i in [0,1,2]:
    print(f'sol_index: {i}\n')
    T = 1
    if source == 'reduced_source':
        f,u,w = functions.manufacture_elasticity_solution(d1=3, d2=2, **functions.ELsols3d[i])
    else:
        f,u,w = functions.manufacture_elasticity_solution(d1=2, d2=2, **functions.ELsols[i])
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=source=='zero_source', name=f'ELsol{i}',w_raw=w)
    
    model = solvers.Solvers(equation='elasticity', modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figname = None
    figname = f'../master/2d_elastic_figures/{source}/{mode}/interpol/loss_sol{i}{extra_tag}.pdf'
    model_folder = None
    model.plot=False
    #model.train(figname=figname, model_folder = model_folder)
    model.train(figname=figname)
    #model.load_weights(model_folder)
    
    figname = None
    figname = f'../master/2d_elastic_figures/{source}/{mode}/interpol/sol{i}{extra_tag}.pdf'
    _ = model.test(interpol = True, figname=figname)
    figname = f'../master/2d_elastic_figures/{source}/{mode}/extrapol/sol{i}{extra_tag}.pdf'
    _ = model.test(interpol = False, figname=figname)
