import numpy as np
#import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import FEM
import functions

COLORS = {
        'DNN':'y',
        'CoSTA_DNN':'g',
        'LSTM':'r',
        'CoSTA_LSTM':'c',
        }

def get_DNN(input_shape, output_length, n, l, lr):
    def lrelu(x):
        return tf.keras.activations.relu(x, alpha=0.01)#, threshold=0,  max_value=0.01)

    model = keras.Sequential(
        [
            layers.Dense(
                n,
                #activation="sigmoid",
                activation=lrelu,
                input_shape=input_shape,
                #kernel_regularizer=keras.regularizers.L2(),
                #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.10, maxval=0.10),
                #kernel_initializer='zeros',
                #bias_initializer='zeros',
                ),
            #layers.LeakyReLU(0.01),
            #layers.ReLU(),#negative_slope = 0.00),
        ] + [
            layers.Dense(
                n,
                #activation="sigmoid",
                activation=lrelu,
                #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
                #kernel_initializer='zeros',
                #bias_initializer='zeros',
                ),
            #layers.LeakyReLU(0.01),
            #layers.ReLU(),#negative_slope = 0.00),
        ]*(l-2) + [
            layers.Dense(
                output_length,
                #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-init, maxval=init) if init else None,
                #kernel_initializer='zeros',
                #bias_initializer=tf.keras.initializers.RandomUniform(minval=-init, maxval=init) if init else None,
                #bias_initializer='zeros',
                ),
        ]
    )
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def get_LSTM(input_shape, output_length, n, l, lr):
    model = keras.Sequential()
    model.add(layers.LSTM(n, activation='relu', input_shape=input_shape, return_sequences=True))
    for i in range(l-2):
        model.add(layers.LSTM(n, activation='relu', return_sequences=True))
    model.add(layers.LSTM(n, activation='relu', return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(layers.Dense(output_length))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def bottle_neck_NN():
    pass


def merge_first_dims(data):
    '''
    Reshapes numpy array from shape (a,b,...) to (a*b,...), such that input[i,j,...]==output[j*input.shape[0]+i,...]'''
    return np.reshape(
            a = data,
            newshape = (data.shape[0]*data.shape[1], *data.shape[2:]),
            order = 'F')


class DNN_solver:
    def __init__(self, Np, tri, T, time_steps, l=3, n=16, epochs=[5000,5000], patience=[20,20], min_epochs=[50,200], lr=1e-5, noise_level=0):
        self.name = 'DNN'
        self.Np = Np
        self.tri = tri
        self.T = T
        self.time_steps = time_steps
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_DNN(n=n, l=l, input_shape=(self.Np,), output_length = Np-2, lr=lr)

    def __prep_data(self, data, set_norm_params):
        data = merge_first_dims(data)
        np.random.shuffle(data)
        X = data[:,:self.Np]
        Y = data[:,self.Np:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,1:-1] += np.random.rand(X.shape[0],self.Np-2)*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data):
        X,Y = self.__prep_data(train_data['pnn'], True)
        X_val,Y_val = self.__prep_data(val_data['pnn'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])


    def __call__(self, f, u_ex, alpha=None):
        plot_steps=False
        if alpha == None:
            alpha = self.alpha_test[0]
        X=np.zeros((1,self.Np))
        u_NN = u_ex(self.tri, t=0)[1:-1]
        k=self.T/self.time_steps
        if plot_steps:
            plt.close()
        for t in range(self.time_steps):
            u_prev = u_NN # save previous solution
            X[:,1:self.Np-1] = u_prev
            X[:,0]=u_ex(x=0,t=t*k)
            X[:,self.Np-1]=u_ex(x=1,t=t*k)
            if plot_steps:
                if t%10==4:
                    print(f'time is {t*k}')
                    plt.plot(self.tri, X[0,:self.Np],'r', label='new')
                    plt.plot(self.tri, u_ex(x=self.tri, t=t*k),'k--', label='ex')
            X = (X-self.mean)/self.var**0.5 # normalize NN input
            u_NN = self.model(X)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
        if plot_steps:
            plt.grid()
            #plt.legend()
            plt.show()
        result = np.zeros((self.Np))
        result[1:self.Np-1] = u_NN
        result[0] = u_ex(x=0,t=self.T)
        result[self.Np-1] = u_ex(x=1,t=self.T)
        return result


class LSTM_solver:
    def __init__(self, Np, tri, T, time_steps, l=3, n=16, epochs=[5000,5000], patience=[20,20], min_epochs=[50,200], lr=1e-5, noise_level=0):
        self.name = 'LSTM'
        self.Np = Np
        self.tri = tri
        self.T = T
        self.time_steps = time_steps
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = 10

        self.model = get_LSTM(n=n, l=l, input_shape=(self.input_period, self.Np,), output_length = Np-2, lr=lr)

    def __prep_data(self, data, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X = np.zeros((data.shape[0]+1-l, data.shape[1], l, self.Np))
        for t in range(1, 1+l):
            X[:,:,l-t,:] = data[l-t:data.shape[0]-t+1, :, :self.Np]
        X = merge_first_dims(X)
        Y = data[l-1:,:, self.Np:]
        Y = merge_first_dims(Y)

        # Shuffle input and output
        P = np.random.permutation(X.shape[0])
        X = X[P,:,:]
        Y = Y[P,:]

        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,:,1:-1] += np.random.rand(X.shape[0],l,self.Np-2)*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data):
        X,Y = self.__prep_data(train_data['pnn'], True)
        X_val,Y_val = self.__prep_data(val_data['pnn'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])


    def __call__(self, f, u_ex, alpha=None):
        l=self.input_period
        X=np.zeros((1,l, self.Np))
        k=self.T/self.time_steps
        for t in range(1,l+1):
            X[:,l-t,:] = u_ex(self.tri, t=-t*k) # TODO: find a porper way to do this (backward exxtrapolation not always possible
        X = (X-self.mean)/self.var**0.5 # normalize NN input (only last entry)
        u_NN = u_ex(self.tri, t=0)[1:-1]
        for t in range(self.time_steps):
            X[:,:-1,:] = X[:,1:,:] # move input data one step forward
            u_prev = u_NN # save previous solution
            X[:,-1:,1:self.Np-1] = u_prev
            X[:,-1,0]=u_ex(x=0,t=t*k)
            X[:,-1,-1]=u_ex(x=1,t=t*k)
            X[:,-1,:] = (X[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            u_NN = self.model(X)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
        result = np.zeros((self.Np))
        result[1:self.Np-1] = u_NN
        result[0] = u_ex(x=0,t=self.T)
        result[self.Np-1] = u_ex(x=1,t=self.T)
        return result

class CoSTA_DNN_solver:
    def __init__(self, Np, T, p, tri, time_steps, l=3, n=16, epochs=[5000,5000], patience=[20,20], min_epochs=[50,200], lr=1e-5, noise_level=0):
        self.name = 'CoSTA_DNN'
        self.Np = Np
        self.p = p
        self.T = T
        self.tri = np.linspace(0,1,self.Np)
        self.time_steps = time_steps
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_DNN(n=n, l=l, input_shape=(self.Np,), output_length = Np-2, lr=lr)

    def __prep_data(self, data, set_norm_params):
        data = merge_first_dims(data)
        np.random.shuffle(data)
        X = data[:,:self.Np]
        Y = data[:,self.Np:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,1:-1] += np.random.rand(X.shape[0],self.Np-2)*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data):
        X,Y = self.__prep_data(train_data['ham'], True)
        X_val,Y_val = self.__prep_data(val_data['ham'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])

    def __call__(self, f, u_ex, alpha=None):
        X=np.zeros((1,self.Np))

        fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
        fem_model.u_fem =fem_model.u_ex(fem_model.tri, t=fem_model.time)
        for t in range(self.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,:self.Np] = fem_step
            X = (X-self.mean)/self.var**0.5
            correction = np.zeros(self.Np)
            correction[1:-1] = self.model(X)[0,:]
            correction[1:-1] = correction[1:-1]*self.Y_var**0.5
            correction[1:-1] = correction[1:-1] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
        return fem_model.u_fem


class CoSTA_LSTM_solver:
    def __init__(self, Np, T, p, tri, time_steps, l=3, n=16, epochs=[5000,5000], patience=[20,20], min_epochs=[50,200], lr=1e-5, noise_level=0):
        self.name = 'CoSTA_LSTM'
        self.Np = Np
        self.p = p
        self.T = T
        self.tri = np.linspace(0,1,self.Np)
        self.time_steps = time_steps
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = 10

        self.model = get_LSTM(n=n, l=l, input_shape=(self.input_period, self.Np,), output_length = Np-2, lr=lr)

    def __prep_data(self, data, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X = np.zeros((data.shape[0]+1-l, data.shape[1], l, self.Np))
        for t in range(1, 1+l):
            X[:,:,l-t,:] = data[l-t:data.shape[0]-t+1, :, :self.Np]
        X = merge_first_dims(X)
        Y = data[l-1:,:, self.Np:]
        Y = merge_first_dims(Y)

        # Shuffle input and output
        P = np.random.permutation(X.shape[0])
        X = X[P,:,:]
        Y = Y[P,:]

        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,:,1:-1] += np.random.rand(X.shape[0],l,self.Np-2)*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data):
        X,Y = self.__prep_data(train_data['ham'], True)
        X_val,Y_val = self.__prep_data(val_data['ham'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])

    def __call__(self, f, u_ex, alpha=None):
        l=self.input_period
        X=np.zeros((1,l, self.Np))
        k=self.T/self.time_steps
        X[:,0,:] = u_ex(self.tri, t=0)

        # Define fem model
        fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)

        # Calculate initial input vector
        for t in range(1,l+1):
            fem_model.u_fem = u_ex(self.tri, t=-t*k) # TODO: find a porper way to do this (backward exxtrapolation not always possible
            fem_model.time = -t*k
            fem_model.step() # make one step from exact solution (as the LSTM model is trained on)
            X[:,l-t,:] = fem_model.u_fem
        X = (X-self.mean)/self.var**0.5 # normalize NN input
  
        # Reset fem model
        fem_model.time = 0
        fem_model.u_fem =fem_model.u_ex(fem_model.tri, t=fem_model.time)

        for t in range(self.time_steps):
            X[:,:-1,:] = X[:,1:,:] # move input data one step forward
            u_prev = fem_model.u_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,-1:,:self.Np] = fem_step
            X[:,-1,:] = (X[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            correction = np.zeros(self.Np)
            correction[1:-1] = self.model(X)[0,:]
            correction[1:-1] = correction[1:-1]*self.Y_var**0.5
            correction[1:-1] = correction[1:-1] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
        return fem_model.u_fem


#####################################################################
###   Solver class compares the different solvers defined above   ###
#####################################################################
class Solvers:
    def __init__(self, models=None, modelnames=None, sol=3, unknown_source=True, Ne=10, time_steps=20, p=1, T=1, **NNkwargs):
        '''
        either models or modelnames must be specified, not both
        models - list of models
        modelnames - dict of names of models to be created, and how many of each
        '''
        assert models == None or modelnames == None
        assert models != None or modelnames != None

        self.sol = sol # index of manufactured SOLution
        self.Ne = Ne
        self.p = p
        self.T = T
        self.Np = Ne*p+1
        self.tri = np.linspace(0,1,self.Np)
        self.time_steps = time_steps
        self.alpha_train = [.1,.2,.3,.4,.5,.6,.9,1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2]
        self.alpha_val = [.8,1.1]#, 1,8, 0.4]
        self.alpha_test_interpol = [.7,1.5]
        self.alpha_test_extrapol = [-0.5,2.5]
        self.plot = True
        self.unknown_source = unknown_source

        if modelnames != None:
            self.modelnames = modelnames
            self.models = []
            for modelname in modelnames:
                for i in range(modelnames[modelname]):
                    if modelname == 'DNN':
                        self.models.append(DNN_solver(T=T, tri=self.tri, time_steps=time_steps, Np=self.Np, **NNkwargs))
                    if modelname == 'LSTM':
                        self.models.append(LSTM_solver(T=T, tri=self.tri, time_steps=time_steps, Np=self.Np, **NNkwargs))
                    if modelname == 'CoSTA_DNN':
                        self.models.append(CoSTA_DNN_solver(p=p, T=T, Np=self.Np, tri=self.tri, time_steps=time_steps, **NNkwargs))
                    if modelname == 'CoSTA_LSTM':
                        self.models.append(CoSTA_LSTM_solver(p=p, T=T, Np=self.Np, tri=self.tri, time_steps=time_steps, **NNkwargs))

        if models != None:
            self.models = models
            self.modelnames = {}
            for model in models:
                if model.name in self.modelnames:
                    self.modelnames[model.name] += 1
                else:
                    self.modelnames[model.name] = 1

        self.create_data()



    def __data_set(self,alphas):
        ham_data=np.zeros((self.time_steps,len(alphas),self.Np + self.Np - 2)) # data for ham NN model
        pnn_data=np.zeros((self.time_steps,len(alphas),self.Np + self.Np - 2)) # data for pure NN model
        extra_feats=np.zeros((self.time_steps,len(alphas),2)) # alpha and time
        for i, alpha in enumerate(alphas):
            print(f'--- making data set for alpha {alpha} ---')
            f, u_ex = functions.sbmfact(T=5, alpha=alpha)[self.sol]
            if self.unknown_source:
                f=FEM.zero
            fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
            for t in range(self.time_steps):
                ex_step = fem_model.u_ex(fem_model.tri, t=fem_model.time) # exact solution before step
                pnn_data[t,i,:self.Np] = ex_step
                fem_model.u_fem = ex_step # Use u_ex as u_prev
                fem_model.step() # make a step
                ham_data[t,i,:self.Np] = fem_model.u_fem
                extra_feats[t,i,0] = alpha
                extra_feats[t,i,1] = fem_model.time
                ex_step = fem_model.u_ex(fem_model.tri, t=fem_model.time) # exact solution after step
                error = ex_step - fem_model.u_fem
                pnn_data[t,i, self.Np:] = ex_step[1:-1]
                ham_data[t,i, self.Np:] = (fem_model.MA @ error)[1:-1] # residual
        return {'ham':ham_data, 'pnn':pnn_data, 'extra_feats':extra_feats}


    def create_data(self):
        start_time = datetime.datetime.now()
        self.train_data = self.__data_set(self.alpha_train)
        self.val_data = self.__data_set(self.alpha_val)

        plot_one_step_sources = False
        if plot_one_step_sources:
            self.test_data = self.__data_set(self.alpha_val) # for plotting learnt source term only !
        print(f'\nTime making data set: {datetime.datetime.now()-start_time}')


    def train(self, figname=None):
        start_time = datetime.datetime.now()

        # prepare plotting
        fig, axs = plt.subplots(1,len(self.modelnames)) # For now, this requires self.models to be sorted by names
        if len(self.modelnames) == 1:
            axs=[axs] # required for indexing later
        prev_name = ''
        i=-1

        for model in self.models:

            # train
            model_start_time = datetime.datetime.now()
            model.train(self.train_data, self.val_data)
            print(f'\nTime training model "{model.name}": {datetime.datetime.now()-model_start_time}')

            # plot history
            if model.name != prev_name:
                prev_name = model.name
                i+=1
                if i>len(self.modelnames):
                    print("Models is not sorted by name") # note testing also needs this
                    1/0
                epochs_vector = np.arange(1, len(model.train_hist)+1)
                axs[i].plot(epochs_vector, model.train_hist, 'b--', label='train_losses')
                axs[i].plot(epochs_vector, model.val_hist, 'r', label='val_losses')
            else:
                epochs_vector = np.arange(1, len(model.train_hist)+1)
                axs[i].plot(epochs_vector, model.train_hist, 'b--')
                axs[i].plot(epochs_vector, model.val_hist, 'r')
            axs[i].set_yscale('log')
            axs[i].set_xlabel('Number of epochs')
            axs[i].set_ylabel('Loss (MSE)')
            axs[i].legend(title=model.name)
            axs[i].grid()

        print(f'\nTime training all models: {datetime.datetime.now()-start_time}')

        if figname != None:
            plt.savefig(figname)
        if self.plot:
            plt.show()
        else:
            plt.close()

        #if plot_one_step_sources:
        #    # Plot some source terms
        #    for t in range(3):
        #        plt.plot(self.tri[1:-1], Y[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.tri[1:-1], model(np.array([X[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='trainset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_train_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.tri[1:-1], Y_val[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.tri[1:-1], model(np.array([X_val[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='valset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_val_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.tri[1:-1], Y_test[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.tri[1:-1], model(np.array([X_test[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='testset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_test_{t}.pdf')
        #        plt.show()

        #        # PNN
        #        plt.plot(self.tri[1:-1], pnnY[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.tri[1:-1], model(np.array([pnnX[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='trainset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_train_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.tri[1:-1], pnnY_val[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.tri[1:-1], model(np.array([pnnX_val[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='valset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_val_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.tri[1:-1], pnnY_test[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.tri[1:-1], model(np.array([pnnX_test[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='testset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_test_{t}.pdf')
        #        plt.show()


    def test(self, interpol = True, figname=None, statplot = 5):
        '''
        interpol - (bool) use interpolating or extrapolating alpha set
        figname - (string or None) filename to save figure to. Note saved if None.
        statplot - (bool or int) if plots should be of mean and variance (instead of every solution). If int, then statplot is set to len(models)>statplot
        '''
        start_time = datetime.datetime.now()
        alphas = self.alpha_test_interpol if interpol else self.alpha_test_extrapol

        # prepare plotting
        if type(statplot) == int:
            statplot = len(self.models) > statplot
        fig, axs = plt.subplots(1,len(alphas))

        for i, alpha in enumerate(alphas):
            f, u_ex = functions.sbmfact(T=self.T,alpha=alpha)[self.sol]
            if self.unknown_source:
                f=FEM.zero

            # Solve with FEM
            fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
            fem_model.solve(self.time_steps, T = self.T)
            tri_fine = np.linspace(0,1,self.Ne*self.p*8+1)
            axs[i].plot(tri_fine, fem_model.solution(tri_fine), 'b', label='fem')

            # prepare plotting
            prev_name = ''
            graphs = {}
            L2_scores = {'FEM' : [fem_model.relative_L2()]}
            def rel_l2() : return np.sqrt(np.mean((fem_model.u_fem-fem_model.u_ex(self.tri,self.T))**2)) / np.sqrt(np.mean(fem_model.u_ex(self.tri,self.T)**2))
            l2_scores = {'FEM' : [rel_l2()]}

            for model in self.models:
                fem_model.u_fem = model(f,u_ex,alpha) # store in fem_model for easy use of relative_L2 and soltion functoins
                if prev_name != model.name:
                    prev_name = model.name
                    graphs[model.name] = []
                    scores[model.name] = []
                    if not statplot:
                        axs[i].plot(tri_fine, fem_model.solution(tri_fine), COLORS[model.name], label=model.name)
                else:
                    if not statplot:
                        axs[i].plot(tri_fine, fem_model.solution(tri_fine), COLORS[model.name])
                graphs[model.name].append(fem_model.solution(tri_fine))
                L2_scores[model.name].append(fem_model.relative_L2())
                l2_scores[model.name].append(rel_l2())

            if statplot:
                for name in graphs:
                    curr_graphs = np.array(graphs[name])
                    mean = np.mean(curr_graphs, axis=0)
                    std = np.std(curr_graphs, axis=0, ddof=1) # reduce one degree of freedom due to mean calculation
                    axs[i].plot(tri_fine, mean, color=COLORS[name])
                    axs[i].fill_between(tri_fine, mean+std, mean-std, color=COLORS[name], alpha = 0.4, label = name)
            axs[i].plot(tri_fine, u_ex(tri_fine), 'k--', label='exact')
            axs[i].grid()
            axs[i].legend(title=f'sol={self.sol},a={alpha}')
        print(f'\nTime testing: {datetime.datetime.now()-start_time}')
        if figname != None:
            plt.savefig(figname)
        if self.plot:
            plt.show()
        else:
            plt.close()
        return L2_scores, l2_scores
