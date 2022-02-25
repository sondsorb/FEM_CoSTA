import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, cm
import sys
import sympy as sp
import utils




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
        Ne = 4
        time_steps = 20
        DNNkwargs = {'n_layers':6,'depth':20, 'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgDNNkwargs = {'n_layers_1':4,'n_layers_2':2,'depth':20,'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}#, 'l1_penalty':0.01}
        LSTMkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
        NoM = 3
        time_delta = 5
    elif mode == 'quick_test':
        Ne = 10
        time_steps = 500
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':8e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':8e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        NoM=4
        time_delta = 0.3 # max 30 steps back
    elif mode == 'full_test':
        Ne = 15
        time_steps = 2500
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20]}
        NoM=10
        time_delta = 0

set_args()


if len(sys.argv)>2:
    p = int(sys.argv[2])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        'pgDNN' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        'CoSTA_pgDNN' : NoM,
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

t = sp.symbols('t')
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
alpha = sp.symbols('alpha')
#u = sp.exp(-t/(3-x**2-y**2-(z-(1+alpha)/4)**2)) # TODO: calculate a realistic u more properly
u =alpha*sp.exp(-3*np.pi**2*t)*sp.cos(np.pi*x)*sp.cos(np.pi*y)*sp.cos(np.pi*z)
u +=sp.exp(-12*np.pi**2*t)*sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y)*sp.cos(2*np.pi*z)
u +=alpha**2*sp.exp(-6*np.pi**2*t)*sp.cos(np.pi*x)*sp.cos(np.pi*y)*sp.cos(2*np.pi*z)
xa,xb,ya,yb = -1,1,-1,1
#u = t + 0.5*alpha*(x**2+y**2)+x
#u = 1+sp.sin(2*np.pi*t+alpha)*sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y) # TODO: calculate a realistic u more properly
#xa,xb,ya,yb = 0,1,0,1

f,u = functions.manufacture_solution(u,t,[x,y,z],alpha, d1=3, d2=2)
sol = functions.Solution(T=1e-2, f_raw=f, u_raw=u, zero_source=False, name=f'bp_tst1', time_delta=time_delta)

model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2, NNkwargs=NNkwargs)
extra_tag = '' # for different names when testing specific stuff
figname = None
figname = f'../master/bp_heat_figures/{mode}/interpol/loss_sol{x}{extra_tag}.pdf'
model_folder = None
model.plot=False
#model.train(figname=figname, model_folder = model_folder)
model.train(figname=figname)
#model.load_weights(model_folder)

model.plot=True

# Interpolation
result_folder = f'../master/saved_results/bp_heat/{mode}{extra_tag}/interpol/'
utils.makefolder(result_folder)
_ = model.test(interpol = True, result_folder=result_folder)
figname = f'../master/bp_heat_figures/{mode}/interpol/sol{x}{extra_tag}.pdf'
model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5)
figname = f'../master/bp_heat_figures/{mode}/interpol/sol{x}_nonstat{extra_tag}.pdf'
model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

# Extrapolation
result_folder = f'../master/saved_results/bp_heat/{mode}{extra_tag}/extrapol/'
utils.makefolder(result_folder)
_ = model.test(interpol = False, result_folder=result_folder)
figname = f'../master/bp_heat_figures/{mode}/extrapol/sol{x}{extra_tag}.pdf'
model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5)
figname = f'../master/bp_heat_figures/{mode}/extrapol/sol{x}_nonstat{extra_tag}.pdf'
model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
