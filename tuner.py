import solvers
import functions

import datetime
import numpy as np

# Configs for tuning:
debug_mode = False
#modelname = 'LSTM'
modelname = 'CoSTA_LSTM'
NoM = 3
T = 5
rate = 0.25 # Rate to change parameters
p=1


if debug_mode:
    Ne = 5
    time_steps = 20
    if modelname=='LSTM':
        nnkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'input_period':5}
    elif modelname=='CoSTA_LSTM':
        nnkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'input_period':5}
    else:
        print('implement this first')
        quit()
    trainkwargs = {'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
    NoM = 2
    time_delta = 5
else:
    Ne = 20
    time_steps = 300
    if modelname=='LSTM' or modelname=='CoSTA_LSTM':
        nnkwargs = {'lstm_layers':4, 'lstm_depth':50, 'dense_layers':1, 'dense_depth':50, 'input_period':10}
    else:
        print('implement this first')
        quit()
    trainkwargs = {'lr':8e-5, 'patience':[20,20], 'epochs':[1000,1000], 'min_epochs':[50,50]}
    NoM = 3
    time_delta = 0.5

Np = Ne+1
tri = np.linspace(0,1,Np)

def output(text):
    print(text)
    with open(f'{modelname}tuner_history{"_db" if debug_mode else ""}.txt', 'a') as f:
        f.write(text)
output(f'\n\n\n{datetime.datetime.now()}\nUsing training parameters: {trainkwargs}\n\n')


def get_models(trainkwargs, nnkwargs):
    models=[]
    kwargs = {**trainkwargs, **nnkwargs}
    for i in range(NoM):
        #if modelname == 'DNN':
        #    models.append(solvers.DNN_solver(T=T, tri=tri, time_steps=time_steps, Np=Np, **DNNkwargs))
        if modelname == 'LSTM':
            models.append(solvers.LSTM_solver(T=T, tri=tri, time_steps=time_steps, Np=Np, **kwargs))
        #if modelname == 'CoSTA_DNN':
        #    models.append(solvers.CoSTA_DNN_solver(T=T, Np=Np, tri=tri, time_steps=time_steps, **DNNkwargs))
        if modelname == 'CoSTA_LSTM':
            models.append(solvers.CoSTA_LSTM_solver(p=p,T=T, Np=Np, tri=tri, time_steps=time_steps, **kwargs))
    return models



# prepare solvers:
f0,u0 = functions.SMBFACT_TUNING[0]
f1,u1 = functions.SMBFACT_TUNING[1]
sol0 = functions.Solution(T=T, f_raw=f0, u_raw=u0, zero_source=True, name=f'T0', time_delta=time_delta)
sol1 = functions.Solution(T=T, f_raw=f1, u_raw=u1, zero_source=True, name=f'T1', time_delta=time_delta)

models0 = get_models(trainkwargs, nnkwargs)
models1 = get_models(trainkwargs, nnkwargs)
s0 = solvers.Solvers(models=models0, sol=sol0, Ne=Ne, time_steps=time_steps)
s1 = solvers.Solvers(models=models1, sol=sol1, Ne=Ne, time_steps=time_steps)
s0.plot = False
s1.plot = False

parameters = [key for key in nnkwargs]


def get_score(nnkwargs, i, tag=''):
    models0 = get_models(trainkwargs, nnkwargs)
    models1 = get_models(trainkwargs, nnkwargs)
    s0.models = models0
    s1.models = models1
    figname = f'../preproject/1d_heat_figures/tunetraining/{i}_{tag}_sol0.pdf'
    s0.train(figname)
    figname = f'../preproject/1d_heat_figures/tunetraining/{i}_{tag}_sol1.pdf'
    s1.train(figname)
    score = np.mean( [
        *s0.test()[1][modelname],
        *s1.test()[1][modelname]
        ])
    return score

# tune:
for tuning_iteration in range(500):
    score = get_score(nnkwargs, tuning_iteration, '')
    text = f'Score to beat: {score}\nConfigs now: {nnkwargs}\n'
    output(text)

    np.random.shuffle(parameters) # start with a random parameter
    total_changes = 0
    for parameter in parameters:
        signs = [-1,1]
        np.random.shuffle(signs) # Start changing in a random direction
        for sign in signs:
            for changes in range(500):
                change = sign * max(1, int(rate*nnkwargs[parameter]))

                text = f'Changing {parameter} from {nnkwargs[parameter]} by adding {change}\n'
                output(text)

                new_nnkwargs = nnkwargs.copy()
                new_nnkwargs[parameter] += change
                new_score = get_score(new_nnkwargs, tuning_iteration, f'{parameter}{change}')

                text = f'new score {new_score}\n'
                output(text)
                
                if new_score < score:
                    nnkwargs = new_nnkwargs
                    score=new_score
                else:
                    break
            if changes > 0:
                break
        total_changes += changes
    if total_changes == 0:
        break

output(f'tuner finished(!), after {tuning_iterations} iterations. Final score: {score}, found with params\n{nnkwargs}\n\n')
