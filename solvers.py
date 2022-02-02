import numpy as np
#import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import json

import FEM
from utils import merge_first_dims

COLORS = {
        # (blac)k is reserved for exact solution
        'FEM':(1,0,0),# blue
        'DNN':(0.8,0.4,0), #gold
        'pgDNN':(1,1,0), # yellow
        'CoSTA_DNN': (0,0,1), # blue
        'CoSTA_pgDNN':(0,0.5,0), # Green
        'LSTM':(.86,0,1), # purple-pink
        'pgLSTM':'maroon',
        'CoSTA_LSTM':'c',
        'CoSTA_pgLSTM':'m',
        }

def lrelu(x):
    return tf.keras.activations.relu(x, alpha=0.01)#, threshold=0,  max_value=0.01)

def get_DNN(input_shape, output_length, n_layers, depth, bn_depth, lr): #TODO remove/change arguments, and use them

    model = keras.Sequential(
        [
            layers.Dense(
                depth,
                activation=lrelu,
                input_shape=input_shape,
                ),
        ] + [
            layers.Dense(
                depth2,#depth, #TODO fix this temporary workaround
                activation=lrelu,
                )
            for depth2 in [bn_depth,depth,depth]] + [ #for i in range(n_layers-3)] + [
            layers.Dense(
                output_length,
                ),
        ]
    )
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    #model.summary()
    return model

def get_pgDNN(input_shape_1, input_shape_2, output_length, n_layers_1, depth, bn_depth, n_layers_2, lr, l1_penalty=0): #TODO: remove/change arguments, and use them
    '''
    Fully connected nerual network with 2 inputs, one at the start and one at a bottleneck
    '''
    L1_reg = keras.regularizers.L1(l1=l1_penalty)

    inputs_1 = keras.Input(shape=input_shape_1)
    x = inputs_1

    for current_depth in [depth,bn_depth]:
        x = layers.Dense(current_depth, kernel_regularizer=L1_reg)(x)
    model_1 = keras.Model(inputs_1, x)

    inputs_2 = keras.Input(shape=input_shape_2)
    x = inputs_2
    model_2 = keras.Model(inputs_2, x)

    combined_input = layers.concatenate([model_1.output, model_2.output])
    x = combined_input 
    for current_depth in [depth,depth,output_length]:
        x = layers.Dense(current_depth, kernel_regularizer=L1_reg)(x)


    model = keras.Model(inputs=[model_1.input, model_2.input], outputs=x)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    #model.summary()
    return model

def get_LSTM(input_shape, output_length, dense_layers, dense_depth, lstm_layers, lstm_depth, lr, dropout_level=0):
    model = keras.Sequential()
    
    # Add correct amount of LSTM layers
    if lstm_layers == 1:
        model.add(layers.LSTM(lstm_depth, activation=lrelu, input_shape=input_shape, return_sequences=False))
    if lstm_layers  > 1:
        model.add(layers.LSTM(lstm_depth, activation=lrelu, input_shape=input_shape, return_sequences=True))
        for i in range(lstm_layers -2):
            model.add(layers.LSTM(lstm_depth, activation=lrelu, return_sequences=True))
        model.add(layers.LSTM(lstm_depth, activation=lrelu, return_sequences=False))
    model.add(layers.Dropout(dropout_level))

    for i in range(dense_layers-1):
        model.add(layers.Dense(dense_depth, activation=lrelu))
    model.add(layers.Dense(output_length))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def bottle_neck_NN():
    pass


class DNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'DNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_DNN(input_shape=(self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)

    def __prep_data(self, data, set_norm_params):
        data = merge_first_dims(data)
        np.random.shuffle(data)
        X = data[:,:self.disc.Nv]
        Y = data[:,self.disc.Nv:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,self.disc.inner_ids2] += np.random.rand(X.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], True)
        if weights_loaded:
            return
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


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        plot_steps=False
        X=np.zeros((1,self.disc.Nv))
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        k=self.disc.k
        if plot_steps:
            plt.close()
        for t in range(self.disc.time_steps):
            u_prev = u_NN # save previous solution
            X[:,self.disc.inner_ids2] = u_prev
            X[:,self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            if plot_steps:
                if t%10==4:
                    print(f'time is {t*k}')
                    plt.plot(self.disc.pts, X[0,:self.disc.Nv],'r', label='new')
                    plt.plot(self.disc.pts, u_ex(x=self.disc.pts, t=t*k),'k--', label='ex')
            X = (X-self.mean)/self.var**0.5 # normalize NN input
            u_NN = self.model(X)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        if plot_steps:
            plt.grid()
            #plt.legend()
            plt.show()
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result


class pgDNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'pgDNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]]
        self.noise_level = noise_level

        self.model = get_pgDNN(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):
        data_1 = merge_first_dims(data_1)
        data_2 = merge_first_dims(data_2)
        P = np.random.permutation(data_1.shape[0])
        data_1 = data_1[P,:]
        X1 = data_1[:,:self.disc.Nv]
        Y = data_1[:,self.disc.Nv:]
        X2 = data_2[P,:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], train_data['extra_feats'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], val_data['extra_feats'], False)

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


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X1=np.zeros((self.disc.Nv))
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        k=self.disc.k
        for t in range(self.disc.time_steps):
            u_prev = u_NN # save previous solution
            X1[self.disc.inner_ids2] = u_prev
            X1[self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            X2 = np.array([alpha,(t+1)*k])
            X1 = (X1-self.mean)/self.var**0.5 # normalize NN input
            X2 = (X2-self.X2_mean)/self.X2_var**0.5 # normalize NN input
            model_input = [np.array([X1]), np.array([X2])]
            u_NN = self.model(model_input)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result


class LSTM_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, input_period=10, **NNkwargs):
        self.name = 'LSTM'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = input_period

        self.model = get_LSTM(input_shape=(self.input_period, self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)

    def __prep_data(self, data, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X = np.zeros((data.shape[0]+1-l, data.shape[1], l, self.disc.Nv))
        for t in range(1, 1+l):
            X[:,:,l-t,:] = data[l-t:data.shape[0]-t+1, :, :self.disc.Nv]
        X = merge_first_dims(X)
        Y = data[l-1:,:, self.disc.Nv:]
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
        X[:,:,self.disc.inner_ids2] += np.random.rand(X.shape[0],l,len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], True)
        if weights_loaded:
            return
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


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        l=self.input_period
        X=np.zeros((1,l, self.disc.Nv))
        k=self.disc.k
        for t in range(1,l+1):
            X[:,l-t,:] = self.disc.format_u(u_ex(self.disc.pts, t=-t*k)) # TODO: find a porper way to do this (backward exxtrapolation not always possible
        X = (X-self.mean)/self.var**0.5 # normalize NN input (only last entry)
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        for t in range(self.disc.time_steps):
            X[:,:-1,:] = X[:,1:,:] # move input data one step forward
            u_prev = u_NN # save previous solution
            X[:,-1:,self.disc.inner_ids2] = u_prev
            X[:,-1,self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            X[:,-1,:] = (X[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            u_NN = self.model(X)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result

class CoSTA_DNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'CoSTA_DNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_DNN(input_shape=(self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)

    def __prep_data(self, data, set_norm_params):
        data = merge_first_dims(data)
        np.random.shuffle(data)
        X = data[:,:self.disc.Nv]
        Y = data[:,self.disc.Nv:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,self.disc.inner_ids2] += np.random.rand(X.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], True)
        if weights_loaded:
            return
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

    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X=np.zeros((1,self.disc.Nv))

        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
        for t in range(self.disc.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,:self.disc.Nv] = fem_step
            X = (X-self.mean)/self.var**0.5
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(X)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem


class CoSTA_pgDNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'CoSTA_pgDNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_pgDNN(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):
        data_1 = merge_first_dims(data_1)
        data_2 = merge_first_dims(data_2)
        P = np.random.permutation(data_1.shape[0])
        data_1 = data_1[P,:]
        X1 = data_1[:,:self.disc.Nv]
        Y = data_1[:,self.disc.Nv:]
        X2 = data_2[P,:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], train_data['extra_feats'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['ham'], val_data['extra_feats'], False)

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


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X1=np.zeros((self.disc.Nv))

        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
        for t in range(self.disc.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X1[:self.disc.Nv] = fem_step
            X1 = (X1-self.mean)/self.var**0.5
            X2 = np.array([alpha,fem_model.time])
            X2 = (X2-self.X2_mean)/self.X2_var**0.5
            model_input = [np.array([X1]), np.array([X2])]
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(model_input)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem


class CoSTA_LSTM_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, input_period=10, **NNkwargs):
        self.name = 'CoSTA_LSTM'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = input_period

        self.model = get_LSTM(input_shape=(self.input_period, self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)

    def __prep_data(self, data, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X = np.zeros((data.shape[0]+1-l, data.shape[1], l, self.disc.Nv))
        for t in range(1, 1+l):
            X[:,:,l-t,:] = data[l-t:data.shape[0]-t+1, :, :self.disc.Nv]
        X = merge_first_dims(X)
        Y = data[l-1:,:, self.disc.Nv:]
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
        X[:,:,self.disc.inner_ids2] += np.random.rand(X.shape[0],l,len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], True)
        if weights_loaded:
            return
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

    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        l=self.input_period
        X=np.zeros((1,l, self.disc.Nv))
        k=self.disc.k
        X[:,0,:] = self.disc.format_u(u_ex(self.disc.pts, t=0))

        # Define fem model
        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)

        # Calculate initial input vector
        for t in range(1,l+1):
            fem_model.u_fem = self.disc.format_u(u_ex(self.disc.pts, t=-t*k)) # TODO: find a porper way to do this (backward exxtrapolation not always possible
            if self.disc.equation == 'elasticity':
                fem_model.w_fem =self.disc.format_u(fem_model.w_ex(self.disc.pts, t=-t*k))
            fem_model.time = -t*k
            fem_model.step() # make one step from exact solution (as the LSTM model is trained on)
            X[:,l-t,:] = fem_model.u_fem
        X = (X-self.mean)/self.var**0.5 # normalize NN input
  
        # Reset fem model
        fem_model.time = 0
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))

        for t in range(self.disc.time_steps):
            X[:,:-1,:] = X[:,1:,:] # move input data one step forward
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,-1:,:self.disc.Nv] = fem_step
            X[:,-1,:] = (X[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(X)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem


#####################################################################
###   Solver class compares the different solvers defined above   ###
#####################################################################
class Solvers:
    def __init__(self, sol, equation='heat', models=None, modelnames=None, Ne=10, time_steps=20, p=1, xa=0, xb=1, ya=0, yb=1, dim=1, NNkwargs={}):
        '''
        either models or modelnames must be specified, not both
        models - list of models
        modelnames - dict of names of models to be created, and how many of each
        '''
        assert models == None or modelnames == None
        assert models != None or modelnames != None

        self.sol = sol # solution
        self.disc = FEM.Disc(
                equation=equation,
                T=sol.T, 
                time_steps=time_steps, 
                Ne=Ne,
                p=p,
                dim=dim,
                xa=xa,
                xb=xb,
                ya=ya,
                yb=yb
                )
        self.alpha_train = [.1,.2,.3,.4,.5,.6,.9,1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2]
        self.alpha_val = [.8,1.1]#, 1,8, 0.4]
        self.alpha_test_interpol = [.7,1.5]
        self.alpha_test_extrapol = [-0.5,2.5]
        self.plot = True

        if modelnames != None:
            self.modelnames = modelnames
            self.models = []
            for modelname in modelnames:
                for i in range(modelnames[modelname]):
                    if modelname == 'DNN':
                        self.models.append(DNN_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'pgDNN':
                        self.models.append(pgDNN_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'LSTM':
                        self.models.append(LSTM_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_DNN':
                        self.models.append(CoSTA_DNN_solver(disc=self.disc,**(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_pgDNN':
                        self.models.append(CoSTA_pgDNN_solver(disc=self.disc,**(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_LSTM':
                        self.models.append(CoSTA_LSTM_solver(disc=self.disc, **(NNkwargs[modelname])))
                    else:
                        print(f'WARNING!!!! model named {modelname} not implemented')

        if models != None:
            self.models = models
            self.modelnames = {}
            for model in models:
                if model.name in self.modelnames:
                    self.modelnames[model.name] += 1
                else:
                    self.modelnames[model.name] = 1

        self.create_data()



    def __data_set(self,alphas, silence=False):
        ham_data=np.zeros((self.disc.time_steps,len(alphas),self.disc.Nv + len(self.disc.inner_ids2))) # data for ham NN model
        pnn_data=np.zeros((self.disc.time_steps,len(alphas),self.disc.Nv + len(self.disc.inner_ids2))) # data for pure NN model
        extra_feats=np.zeros((self.disc.time_steps,len(alphas),2)) # alpha and time
        for i, alpha in enumerate(alphas):
            if not silence:
                print(f'--- making data set for alpha {alpha} ---')
            self.sol.set_alpha(alpha)
            fem_model = self.disc.make_model(self.sol.f, self.sol.u, w_ex=self.sol.w)
            for t in range(self.disc.time_steps):
                ex_step = self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time)) # exact solution before step
                pnn_data[t,i,:self.disc.Nv] = ex_step
                fem_model.u_fem = ex_step # Use u_ex as u_prev
                if self.disc.equation == 'elasticity':
                    if fem_model.time == 0:
                        fem_model.w_fem = self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
                    else:
                        fem_model.w_fem = self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time)-fem_model.u_ex(fem_model.pts, t=fem_model.time-fem_model.k))/fem_model.k
                fem_model.step() # make a step
                ham_data[t,i,:self.disc.Nv] = fem_model.u_fem
                extra_feats[t,i,0] = alpha
                extra_feats[t,i,1] = fem_model.time
                ex_step = self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time)) # exact solution after step
                error = ex_step - fem_model.u_fem
                pnn_data[t,i, self.disc.Nv:] = ex_step[self.disc.inner_ids2]
                if self.disc.equation == 'elasticity':
                    ham_data[t,i, self.disc.Nv:] = (fem_model.MA @ np.concatenate([error,error/fem_model.k]))[self.disc.inner_ids2] # residual
                else:
                    ham_data[t,i, self.disc.Nv:] = (fem_model.MA @ error)[self.disc.inner_ids2] # residual
        return {'ham':ham_data, 'pnn':pnn_data, 'extra_feats':extra_feats}


    def create_data(self, silence = False):
        start_time = datetime.datetime.now()
        self.train_data = self.__data_set(self.alpha_train, silence)
        self.val_data = self.__data_set(self.alpha_val, silence)

        plot_one_step_sources = False
        if plot_one_step_sources:
            self.test_data = self.__data_set(self.alpha_val) # for plotting learnt source term only !
        print(f'\nTime making data set: {datetime.datetime.now()-start_time}')

    def load_weights(self, model_folder):
        for j, model in enumerate(self.models):
            model.model.load_weights(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}')
            model.train(self.train_data, self.val_data, True) # For setting mean/var, it wont fit the model
        print('model weights loaded')

    def train(self, model_folder=None, figname=None):
        start_time = datetime.datetime.now()

        # prepare plotting
        prev_name = ''
        i=-1

        for j, model in enumerate(self.models):

            # train
            model_start_time = datetime.datetime.now()
            model.train(self.train_data, self.val_data)
            if model_folder != None:
                model.model.save_weights(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}')
            print(f'\nTime training model "{model.name}": {datetime.datetime.now()-model_start_time}')

            # plot history
            if model.name != prev_name:
                prev_name = model.name
                i+=1
                if i>len(self.modelnames):
                    print("Models is not sorted by name") # note testing also needs this
                    1/0
                epochs_vector = np.arange(1, len(model.train_hist)+1)
                plt.plot(epochs_vector, model.train_hist, '--', color=COLORS[model.name], label=f'{model.name},train')
                plt.plot(epochs_vector, model.val_hist, color=COLORS[model.name], label=f'{model.name},val')
            else:
                epochs_vector = np.arange(1, len(model.train_hist)+1)
                plt.plot(epochs_vector, model.train_hist, '--', color=COLORS[model.name])
                plt.plot(epochs_vector, model.val_hist, color=COLORS[model.name])

            if model_folder != None:
                # save history json file
                with open(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}.json', "w") as f:
                    f.write(json.dumps({
                        'train':model.train_hist, 
                        'val':model.val_hist
                        }))

            plt.yscale('log')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend(title='losses')
            plt.grid()

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
        #        plt.plot(self.disc.pts[self.disc.inner_ids1], Y[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([X[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='trainset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_train_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], Y_val[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([X_val[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='valset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_val_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], Y_test[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([X_test[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='testset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_test_{t}.pdf')
        #        plt.show()

        #        # PNN
        #        plt.plot(self.disc.pts[self.disc.inner_ids1], pnnY[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([pnnX[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='trainset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_train_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], pnnY_val[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([pnnX_val[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='valset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_val_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], pnnY_test[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([pnnX_test[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='testset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_test_{t}.pdf')
        #        plt.show()


    def test(self, interpol = True, figname=None, statplot = 5, ignore_models=[]):
        '''
        interpol - (bool) use interpolating or extrapolating alpha set
        figname - (string or None) filename to save figure to. Note saved if None.
        statplot - (bool or int) if plots should be of mean and variance (instead of every solution). If int, then statplot is set to len(models)>statplot
        ignore_models - (list of str) names of models not to include in plots. FEM always included.
        '''
        start_time = datetime.datetime.now()
        alphas = self.alpha_test_interpol if interpol else self.alpha_test_extrapol

        # prepare plotting
        figsize = (6,4)
        if type(statplot) == int:
            statplot = len(self.models) > statplot
        fig, axs = plt.subplots(1,len(alphas), figsize=figsize)

        l2_devs={}
        for i, alpha in enumerate(alphas):
            self.sol.set_alpha(alpha)

            # Define callback function to store l2 error development
            def relative_l2_callback(t, u_approx):
                inner_u = u_approx[self.disc.inner_ids2] if len(u_approx)==self.disc.Nv else u_approx[0,:]
                self.l2_development.append(np.sqrt(np.mean((inner_u-self.disc.format_u(self.sol.u(self.disc.pts[self.disc.inner_ids1],t)))**2)) / np.sqrt(np.mean(self.disc.format_u(self.sol.u(self.disc.pts[self.disc.inner_ids1],t)**2))))

            # Solve with FEM
            self.l2_development = []
            fem_model = self.disc.make_model(self.sol.f, self.sol.u, w_ex=self.sol.w)
            fem_model.solve(self.disc.time_steps, T = self.disc.T, callback = relative_l2_callback)
            if self.disc.dim==1:
                axs[i].plot(self.disc.pts_line, fem_model.solution(self.disc.pts_line), color=COLORS['FEM'], label='fem')
            if self.disc.dim==2:
                if self.disc.equation == 'elasticity':
                    axs[i].plot(self.disc.pts_line[:,0], fem_model.solution(self.disc.pts_line)[:,0], color=COLORS['FEM'], label='fem')
                else:
                    axs[i].plot(self.disc.pts_line[:,0], fem_model.solution(self.disc.pts_line), color=COLORS['FEM'], label='fem')

            # prepare plotting
            prev_name = ''
            graphs = {}
            L2_scores = {'FEM' : [fem_model.relative_L2()]}
            def rel_l2() : return np.sqrt(np.mean((fem_model.u_fem-self.disc.format_u(fem_model.u_ex(self.disc.pts,self.disc.T)))**2)) / np.sqrt(np.mean(self.disc.format_u(fem_model.u_ex(self.disc.pts,self.disc.T)**2)))
            l2_scores = {'FEM' : [rel_l2()]}
            l2_devs[alpha]= {'FEM' : [self.l2_development]}
            
            for model in self.models:
                if not model.name in ignore_models:

                    self.l2_development = []
                    fem_model.u_fem = model(self.sol.f,self.sol.u,alpha,callback=relative_l2_callback, w_ex=self.sol.w) # store in fem_model for easy use of relative_l2 and soltion functoins
                    if prev_name != model.name:
                        prev_name = model.name
                        graphs[model.name] = []
                        L2_scores[model.name] = []
                        l2_scores[model.name] = []
                        l2_devs[alpha][model.name] = []
                        if not statplot:
                            if self.disc.dim==1:
                                axs[i].plot(self.disc.pts_line, fem_model.solution(self.disc.pts_line), color=COLORS[model.name], label=model.name)
                            if self.disc.dim==2:
                                axs[i].plot(self.disc.pts_line[:,0], fem_model.solution(self.disc.pts_line), color=COLORS[model.name], label=model.name)
                    else:
                        if not statplot:
                            if self.disc.dim==1:
                                axs[i].plot(self.disc.pts_line, fem_model.solution(self.disc.pts_line), color=COLORS[model.name])
                            if self.disc.dim==2:
                                axs[i].plot(self.disc.pts_line[:,0], fem_model.solution(self.disc.pts_line), color=COLORS[model.name])
                    if self.disc.equation == 'elasticity':
                        graphs[model.name].append(fem_model.solution(self.disc.pts_line)[:,0])
                    else:
                        graphs[model.name].append(fem_model.solution(self.disc.pts_line))
                    L2_scores[model.name].append(fem_model.relative_L2())
                    l2_scores[model.name].append(rel_l2())
                    l2_devs[alpha][model.name].append(self.l2_development)

            if statplot:
                for name in graphs:
                    curr_graphs = np.array(graphs[name])
                    mean = np.mean(curr_graphs, axis=0)
                    std = np.std(curr_graphs, axis=0, ddof=1) # reduce one degree of freedom due to mean calculation
                    if self.disc.dim==1:
                        axs[i].plot(self.disc.pts_line, mean, color=COLORS[name])
                        axs[i].fill_between(self.disc.pts_line, mean+std, mean-std, color=COLORS[name], alpha = 0.4, label = name)
                    if self.disc.dim==2:
                        axs[i].plot(self.disc.pts_line[:,0], mean, color=COLORS[name])
                        axs[i].fill_between(self.disc.pts_line[:,0], mean+std, mean-std, color=COLORS[name], alpha = 0.4, label = name)
            if self.disc.dim==1:
                axs[i].plot(self.disc.pts_line, self.sol.u(self.disc.pts_line), 'k--', label='exact')
            if self.disc.dim==2:
                if self.disc.equation == 'elasticity':
                    axs[i].plot(self.disc.pts_line[:,0], self.sol.u(self.disc.pts_line)[:,0], 'k--', label='exact')
                else:
                    axs[i].plot(self.disc.pts_line[:,0], self.sol.u(self.disc.pts_line), 'k--', label='exact')
            axs[i].grid()
            axs[i].legend(title=f'sol={self.sol.name},a={alpha}')
        print(f'\nTime testing: {datetime.datetime.now()-start_time}')
        plt.tight_layout()
        if figname != None:
            plt.savefig(figname)
        if self.plot:# and self.disc.dim==1:
            plt.show()
        else:
            plt.close()

        # Plot l2 development:
        if statplot: # not implemented otherwise
            fig, axs = plt.subplots(1,len(alphas), figsize=figsize)
            for i, alpha in enumerate(alphas):
                axs[i].set_yscale('log')
                for name in l2_devs[alpha]:
                    curr_devs = np.array(l2_devs[alpha][name])
                    mean = np.mean(curr_devs, axis=0)
                    if name!='FEM': # TODO use length instead, to open for nonstatplot
                        axs[i].plot(np.arange(len(mean)), mean, color=COLORS[name])
                        std = np.std(curr_devs, axis=0, ddof=1) # reduce one degree of freedom due to mean calculation
                        axs[i].fill_between(np.arange(len(mean)), mean+std, mean, color=COLORS[name], alpha = 0.4, label = name)
                    else:
                        axs[i].plot(np.arange(len(mean)), mean, color=COLORS[name], label=name)

                axs[i].grid()
                axs[i].legend(title=f'sol={self.sol.name},a={alpha}')

            plt.tight_layout()
            if figname != None:
                plt.savefig(figname[:-4]+'_devs'+'.pdf') # TODO do this in a cleaner way
            if self.plot:
                plt.show()
            else:
                plt.close()



        print('scores for last alpha (L2, l2):',L2_scores, l2_scores) # TODO? print earlier for both alphas
        return L2_scores, l2_scores
