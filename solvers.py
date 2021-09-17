import numpy as np
#import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import FEM
import functions




class Costa:

    def __init__(self, sol=3, Ne=10, time_steps=20, p=1, T=1):
        self.sol = sol # index of manufactured SOLution
        self.Ne = Ne
        self.p = p
        self.T = T
        self.Np = 10*p+1
        self.tri = np.linspace(0,1,self.Np)
        self.time_steps = time_steps
        self.alpha_train = [0.1, 0.2] # temporary!TODO
        self.alpha_test = [0.3] # temporary TODO
        
        model1 = keras.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(self.Np+1,), kernel_regularizer=keras.regularizers.L2()),
                layers.Dense(16, activation="relu"),
                layers.Dense(self.Np),
            ]
        )
        opt = keras.optimizers.Adam()#learning_rate=0.0001)
        model1.compile(loss='mse', optimizer=opt)
        
        self.NN_models = {'model1':model1}




    def train(self, plot_history=False):
        X=np.zeros((self.time_steps*len(self.alpha_train),self.Np+1))
        Y=np.zeros((self.time_steps*len(self.alpha_train),self.Np))
        for i in range(len(self.alpha_train)):
            f, u_ex = functions.sbmfact(alpha=self.alpha_train[i])[self.sol]
            fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
            for t in range(self.time_steps):
                fem_model.u_fem = fem_model.u_ex(fem_model.tri, t=fem_model.time) # Use u_ex as u_prev
                fem_model.step() # make a step
                X[i*self.time_steps+t,:-1] = fem_model.u_fem
                X[i*self.time_steps+t,-1] = self.alpha_train[i] # more/less features could be tried
                error = fem_model.u_fem - fem_model.u_ex(fem_model.tri, t=fem_model.time)
                Y[i*self.time_steps+t, :] = fem_model.MA @ error # residual

        # TODO: shuffle & normalize.

        history={}
        for modelname in self.NN_models:
            train_result = self.NN_models[modelname].fit(X, Y, 
                    epochs=50, # TODO
                    #batch_size=512, 
                    #validation_data=(z_val, o_val),
                    #verbose=0
                    )
            history[modelname] = train_result.history
            if plot_history:
                epochs_vector = np.arange(1, len(history[modelname]['loss'])+1)
                plt.plot(epochs_vector, history[modelname]['loss'], '--', label=modelname+'loss')
                #plt.plot(epochs_vector, history[modelname]['val_loss'], label=modelname+'val_loss')
        if plot_history:
            plt.ylim(0)
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid()
            plt.show()


    def __call__(self):
        pass
