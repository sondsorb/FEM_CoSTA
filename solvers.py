import numpy as np
#import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import FEM
import functions




class Costa:

    def __init__(self, sol=2, Ne=10, time_steps=20, p=1, T=1, **NNkwargs):
        self.sol = sol # index of manufactured SOLution
        self.Ne = Ne
        self.p = p
        self.T = T
        self.Np = Ne*p+1
        self.tri = np.linspace(0,1,self.Np)
        #print(self.tri)
        self.time_steps = time_steps
        self.alpha_train = [.1,.2,.3,.4,.5,.6,.9,1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2]
        self.alpha_val = [.8,1.1, 1,8, 0.4]
        self.alpha_test = [.7,1.5] # Interpolation for now
        self.plot = True
        self.unknown_source = True

        self.set_NN_model(**NNkwargs)

    def set_NN_model(self, model=None, l=3, n=16, epochs=10000, patience=20, lr=1e-5):
        self.normalize =True#False
        self.alpha_feature = True
        self.nfeats = 0#1 # number of extra features (like alpha, time,)
        if self.alpha_feature:
            self.nfeats +=1
        self.epochs = epochs
        self.patience = patience
        if model != None:
            self.model=model
            return

        model1 = keras.Sequential(
            [
                layers.Dense(
                    n, 
                    activation="sigmoid", 
                    input_shape=(self.Np+self.nfeats,), 
                    #kernel_regularizer=keras.regularizers.L2(),
                    #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.10, maxval=0.10),
                    #kernel_initializer='zeros',
                    #bias_initializer='zeros',
                    ),
                #layers.LeakyReLU(0.01),
            ] + [
                layers.Dense(
                    n, 
                    activation="sigmoid",
                    #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
                    #kernel_initializer='zeros',
                    #bias_initializer='zeros',
                    ),
                #layers.LeakyReLU(0.01),
            ]*(l-2) + [
                layers.Dense(
                    self.Np-2,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
                    #kernel_initializer='zeros',
                    bias_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
                    #bias_initializer='zeros',
                    ),
            ]
        )
        opt = keras.optimizers.Adam(learning_rate=lr)
        model1.compile(loss='mse', optimizer=opt)
        self.NN_model = model1 # move this out (remember it relies on self.Np)

        


    def __data_set(self,alphas, set_norm_params=False):
        data=np.zeros((self.time_steps*len(alphas),self.Np+self.nfeats + self.Np))
        for i, alpha in enumerate(alphas):
            f, u_ex = functions.sbmfact(alpha=alpha)[self.sol]
            if self.unknown_source:
                f=FEM.zero
            fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
            for t in range(self.time_steps):
                fem_model.u_fem = fem_model.u_ex(fem_model.tri, t=fem_model.time) # Use u_ex as u_prev
                fem_model.step() # make a step
                data[i*self.time_steps+t,:self.Np] = fem_model.u_fem
                if self.alpha_feature:
                    data[i*self.time_steps+t,self.Np:self.Np+self.nfeats] = alpha # more/less features could be tried
                error = fem_model.u_ex(fem_model.tri, t=fem_model.time) - fem_model.u_fem
                data[i*self.time_steps+t, self.Np+self.nfeats:] = fem_model.MA @ error # residual
        np.random.shuffle(data)
        X = data[:,:self.Np+self.nfeats]
        Y = data[:,self.Np+self.nfeats+1:-1]
        #print(X)
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            print('mean, var = ',self.mean, self.var)
        X = X-self.mean
        X = X/self.var**0.5
        #print(X)
        return X,Y



    def train(self):

        start_time = datetime.datetime.now()
        X, Y = self.__data_set(self.alpha_train, set_norm_params=True)
        X_val, Y_val = self.__data_set(self.alpha_val)
        print(f'\nTime making data set: {datetime.datetime.now()-start_time}')

        start_time = datetime.datetime.now()
        train_result = self.NN_model.fit(X, Y, 
                epochs=self.epochs,
                batch_size=32, 
                callbacks=[keras.callbacks.EarlyStopping(patience=self.patience)],
                validation_data=(X_val, Y_val),
                verbose=0
                )
        self.history = train_result.history
        print(f'\nTime training: {datetime.datetime.now()-start_time}')
        if self.plot:
            epochs_vector = np.arange(1, len(self.history['loss'])+1)
            plt.plot(epochs_vector, self.history['loss'], '--', label='loss')
            plt.plot(epochs_vector, self.history['val_loss'], label='val_loss')
        if self.plot:
            #plt.ylim(0, self.history['loss'][4])
            plt.yscale('log')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid()
            plt.show()


    def __call__(self, alpha=None):
        if alpha == None:
            alpha = self.alpha_test[0]
        X=np.zeros((1,self.Np+self.nfeats))
        if self.alpha_feature:
            X[:,-1] = alpha
        f, u_ex = functions.sbmfact(alpha=alpha)[self.sol]
        if self.unknown_source:
            f=FEM.zero
        fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
        fem_model.u_fem =fem_model.u_ex(fem_model.tri, t=fem_model.time)
        for t in range(self.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,:self.Np] = fem_step
            #X = (X-self.mean)/self.var**0.5
            #print(X)
            #print((X-self.mean)/self.var**0.5)
            correction = self.NN_model(X)[0,:].numpy() #* 0.5#.01
            #print('\ncorrecton:\n', correction)
            u_prev[1:-1] = u_prev[1:-1] + correction # add correction to previous solution
            fem_model.u_fem = u_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step() # corrected step
        return fem_model.u_fem


    def test(self, use_validation_set = True):
        start_time = datetime.datetime.now()
        alphas = self.alpha_val if use_validation_set else self.alpha_test
        fem_score = {}
        fem_score_tri = {}
        costa_score = {}
        costa_score_tri = {}
        if self.plot:
            fig, axs = plt.subplots(1,len(alphas))
        for i, alpha in enumerate(alphas):
            f, u_ex = functions.sbmfact(T=self.T,alpha=alpha)[self.sol]
            if self.unknown_source:
                f=FEM.zero
            fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
            fem_model.solve(self.time_steps, T = self.T)
            if self.plot:
                tri_fine = np.linspace(0,1,self.Ne*self.p*8+1)
                axs[i].plot(tri_fine, fem_model.solution(tri_fine), label='fem')

            fem_score[f'{alpha}'] = fem_model.relative_L2()
            fem_score_tri[f'{alpha}'] = np.sqrt(np.mean((fem_model.u_fem-fem_model.u_ex(self.tri,self.T))**2)) / np.sqrt(np.mean(fem_model.u_ex(self.tri,self.T)**2))
            fem_model.u_fem =self(alpha)
            if self.plot:
                axs[i].plot(tri_fine, fem_model.solution(tri_fine), label='costa')
                axs[i].plot(tri_fine, u_ex(tri_fine), '--', label='exact')
                axs[i].grid()
                axs[i].legend(title=f'alpha={alpha}')
            costa_score[f'{alpha}'] = fem_model.relative_L2()
            costa_score_tri[f'{alpha}'] = np.sqrt(np.mean((fem_model.u_fem-fem_model.u_ex(self.tri,self.T))**2)) / np.sqrt(np.mean(fem_model.u_ex(self.tri,self.T)**2))
        print(f'\nTime testing: {datetime.datetime.now()-start_time}')
        print('')
        print('FEM L2 errors:', fem_score)
        print('CoSTA L2 errors:', costa_score)
        print('FEM l2 errors:', fem_score_tri)
        print('CoSTA l2 errors:', costa_score_tri)
        print('')
        if self.plot:
            plt.show()
        return fem_score, costa_score
