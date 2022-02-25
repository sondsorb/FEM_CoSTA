import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
import sys
import utils

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
    global Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta
    if mode == 'bugfix':
        Ne = 5
        time_steps = 20
        DNNkwargs = {'n_layers':6,'depth':20, 'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgDNNkwargs = {'n_layers_1':4,'n_layers_2':2,'depth':20,'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}#, 'l1_penalty':0.01}
        LSTMkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgLSTMkwargs = {'lstm_layers':2, 'lstm_depth':8, 'dense_layers':1, 'dense_depth':80, 'lr':5e-3, 'patience':[10,10], 'input_period':10}
        NoM = 2
        time_delta = 5
    elif mode == 'quick_test':
        Ne = 20
        time_steps = 500
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':8e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':8e-5,'patience':[20,20]}
        #LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':3, 'lstm_depth':40, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        #pgLSTMkwargs = {'lstm_layers':4, 'lstm_depth':16, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20], 'input_period':10}
        pgLSTMkwargs = {'lstm_layers':2, 'lstm_depth':16, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20], 'input_period':10}
        NoM=3
        time_delta = 0.3 # max 30 steps back
    elif mode == 'full_test':
        Ne = 20
        time_steps = 5000
        DNNkwargs = {'n_layers':6,'depth':80, 'bn_depth':8, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'depth':80,'bn_depth':8,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20]}
        pgLSTMkwargs = {'lstm_layers':4, 'lstm_depth':16, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20], 'input_period':10}
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
        source = False
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
        'pgDNN' : NoM,
        'LSTM' : NoM,
        'pgLSTM' : NoM,
        'CoSTA_DNN' : NoM,
        'CoSTA_pgDNN' : NoM,
        'CoSTA_LSTM' : NoM,
        'CoSTA_pgLSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs, 
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs, 
        'CoSTA_LSTM':LSTMkwargs, 
        'pgLSTM':pgLSTMkwargs, 
        'CoSTA_pgLSTM':pgLSTMkwargs, 
        }

for sol_index in [1,0]:
    print(f'sol_index: {sol_index}\n')
    f,u = functions.SBMFACT_TUNING[sol_index]
    sol = functions.Solution(T=5, f_raw=f, u_raw=u, zero_source=not source, name=f'{sol_index}_T', time_delta=time_delta)
    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/loss_sol{sol_index}{extra_tag}.pdf'
    #figname = None
    model.plot=False
    model_folder = f'../master/saved_models/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/'
    #model.train(figname=figname, model_folder = model_folder)
    #model.train(figname=figname)
    model.load_weights(model_folder)

    # Interpolation
    result_folder = f'../master/saved_results/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/sol{sol_index}{extra_tag}.pdf'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/sol{sol_index}_nonstat{extra_tag}.pdf'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../master/saved_results/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/extrapol/sol{sol_index}{extra_tag}.pdf'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/extrapol/sol{sol_index}_nonstat{extra_tag}.pdf'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
