import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
import sys
import utils
import parameters

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

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=1)

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
    model.plot=True
    model_folder = f'../master/saved_models/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/'
    model.train(figname=figname, model_folder = model_folder)
    #model.train(figname=figname)
    #model.load_weights(model_folder)

    # Interpolation
    result_folder = f'../master/saved_results/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/sol{sol_index}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/sol{sol_index}_nonstat{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../master/saved_results/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/extrapol/sol{sol_index}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5)
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/extrapol/sol{sol_index}_nonstat{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
