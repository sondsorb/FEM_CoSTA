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

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=2)

if len(sys.argv)>2:
    p = int(sys.argv[2])
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

class Fem_method:
    # class for making the 2d fem model act as any other model in methods, for solvers workaround with 2 fem models
    def __init__(self, disc, sol_1, sol_2, pts_1d):
        self.disc = disc
        self.sol_1 = sol_1
        self.sol_2 = sol_2
        self.pts_1d = pts_1d
        pts_2d_line = disc.pts[:len(pts_1d)]
        assert (pts_2d_line[:,0] == pts_1d).all()
        self.name = 'FEM_2'
    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        self.sol_2.set_alpha(self.sol_1.alpha)
        self.fem_model = self.disc.make_model(self.sol_2.f, self.sol_2.u, w_ex=self.sol_2.w)
        def new_callback(t,u):
            u_line = u[:len(self.pts_1d)]
            return callback(t,u_line)
        self.fem_model.solve(self.disc.time_steps, T = self.disc.T, callback = new_callback)
        return self.fem_model.u_fem[:len(self.pts_1d)]


xa,xb,ya,yb = 0,1,0,1
for i in [1]:
    f_1,u_1,f_2,u_2 = functions.lowdimred[i]
    sol_1 = functions.Solution(T=1, f_raw=f_1, u_raw=u_1, zero_source=False, name=f'DR_{i}', time_delta=time_delta)
    sol_2 = functions.Solution(T=1, f_raw=f_2, u_raw=u_2, zero_source=False, name=f'DR_{i}', time_delta=time_delta)

    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol_1, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=1, skip_create_data=False, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figfolder = f'../master/bp_heat_figures/lowdr/{mode}/extrapol/'
    utils.makefolder(figfolder)
    figfolder = f'../master/bp_heat_figures/lowdr/{mode}/interpol/'
    utils.makefolder(figfolder)
    figname = f'{figfolder}loss_sol{i}{extra_tag}'
    model_folder = f'../master/saved_models/bp_heat/lowdr/{mode}{extra_tag}/interpol/'
    model.plot=True
    #model.train(figname=figname, model_folder = model_folder)
    model.load_weights(model_folder)

    #ignore_models = ['pgDNN', 'CoSTA_pgDNN']
    ignore_models = []
    legend=True

    # Add 2d FEM model to models
    disc_2d = FEM.Disc(
                equation=model.disc.equation,
                T=sol_2.T, 
                time_steps=model.disc.time_steps, 
                Ne=Ne,
                p=model.disc.p,
                dim=2,
                xa=xa,
                xb=xb,
                ya=ya,
                yb=yb,
                static=model.disc.static,
                )
    fem_method_2d = Fem_method(disc=disc_2d, sol_1=sol_1, sol_2=sol_2, pts_1d=model.disc.pts)
    model.models.append(fem_method_2d)
    model.modelnames[fem_method_2d.name] = 1

    # Interpolation
    result_folder = f'../master/saved_results/bp_heat/lowdr/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/bp_heat_figures/lowdr/{mode}/interpol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5, ignore_models = ignore_models, legend=legend)
    #figname = f'../master/bp_heat_figures/{mode}/interpol/sol{x}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../master/saved_results/bp_heat/lowdr/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/bp_heat_figures/lowdr/{mode}/extrapol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5, ignore_models = ignore_models, legend=legend)
    #figname = f'../master/bp_heat_figures/{mode}/extrapol/sol{x}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
