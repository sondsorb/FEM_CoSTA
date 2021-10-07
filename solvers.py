import numpy as np
#import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import FEM
import functions




class Solvers:

    def __init__(self, sol=3, unknown_source=True, Ne=10, time_steps=20, p=1, T=1, NoM=2, **NNkwargs):
        self.sol = sol # index of manufactured SOLution
        self.Ne = Ne
        self.p = p
        self.T = T
        self.NoM = NoM # Number og (neural network) models (NoM hamNNs + NoM pNNs)
        self.Np = Ne*p+1
        self.tri = np.linspace(0,1,self.Np)
        #print(self.tri)
        self.time_steps = time_steps
        self.alpha_train = [.1,.2,.3,.4,.5,.6,.9,1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2]
        self.alpha_val = [.8,1.1]#, 1,8, 0.4]
        self.alpha_test_interpol = [.7,1.5]
        self.alpha_test_extrapol = [-0.5,2.5]
        self.plot = True
        self.unknown_source = unknown_source

        self.hamNNs= []
        self.pureNNs=[]
        for i in range(NoM):
            self.hamNNs.append(self.set_NN_model(**NNkwargs))#,init=0.05))
            self.pureNNs.append(self.set_NN_model(**NNkwargs,init=False))

    def set_NN_model(self, model=None, l=3, n=16, epochs=10000, patience=[50,50], min_epochs=[250,500], lr=1e-5, noice_level=0, init=False):
        self.normalize =0#True#False
        self.alpha_feature = 0#True
        self.nfeats = 0#1 # number of extra features (like alpha, time,)
        if self.alpha_feature:
            self.nfeats +=1
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noice_level = 1e-2#noice_level
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
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-init, maxval=init) if init else None,
                    #kernel_initializer='zeros',
                    bias_initializer=tf.keras.initializers.RandomUniform(minval=-init, maxval=init) if init else None,
                    #bias_initializer='zeros',
                    ),
            ]
        )
        opt = keras.optimizers.Adam(learning_rate=lr)
        model1.compile(loss='mse', optimizer=opt)
        return model1# move this out (remember it relies on self.Np)

        


    def __data_set(self,alphas, set_norm_params=False):
        ham_data=np.zeros((self.time_steps*len(alphas),self.Np+self.nfeats + self.Np)) # data for ham NN model
        pnn_data=np.zeros((self.time_steps*len(alphas),self.Np+self.nfeats + self.Np)) # data for pure NN model
        for i, alpha in enumerate(alphas):
            print(f'--- making data set for alpha {alpha} ---')
            f, u_ex = functions.sbmfact(alpha=alpha)[self.sol]
            if self.unknown_source:
                f=FEM.zero
            fem_model = FEM.Heat(self.tri, f, self.p, u_ex, k=self.T/self.time_steps)
            for t in range(self.time_steps):
                ex_step = fem_model.u_ex(fem_model.tri, t=fem_model.time) # exact solution before step
                ##prev_ex_step = fem_model.u_ex(fem_model.tri, t=fem_model.time-fem_model.k) ## diff_test
                pnn_data[i*self.time_steps+t,:self.Np] = ex_step
                ##pnn_data[i*self.time_steps+t,:self.Np] = ex_step - prev_ex_step ## diff_test
                ##pnn_data[i*self.time_steps+t,0] = fem_model.u_ex(0, t=fem_model.time+fem_model.k) ## bdry_shift_test
                ##pnn_data[i*self.time_steps+t,self.Np-1] = fem_model.u_ex(1, t=fem_model.time+fem_model.k) ## bdry_shift_test
                fem_model.u_fem = ex_step # Use u_ex as u_prev
                fem_model.step() # make a step
                ham_data[i*self.time_steps+t,:self.Np] = fem_model.u_fem
                if self.alpha_feature:
                    ham_data[i*self.time_steps+t,self.Np:self.Np+self.nfeats] = alpha # more/less features could be tried
                    pnn_data[i*self.time_steps+t,self.Np:self.Np+self.nfeats] = alpha # more/less features could be tried
                ex_step = fem_model.u_ex(fem_model.tri, t=fem_model.time) # exact solution after step
                error = ex_step - fem_model.u_fem
                pnn_data[i*self.time_steps+t, self.Np+self.nfeats:] = ex_step
                ham_data[i*self.time_steps+t, self.Np+self.nfeats:] = fem_model.MA @ error # residual
                #if t%100==4:
                #    print('err and res')
                #    print(error)
                #    #print(fem_model.MA)
                #    print(ham_data[i*self.time_steps+t, self.Np+self.nfeats:])
                #    print(fem_model.F)
                #    print(fem_model.k)
                #    print(fem_model.T)
        np.random.shuffle(ham_data)
        np.random.shuffle(pnn_data)
        X = ham_data[:,:self.Np+self.nfeats]
        Y = ham_data[:,self.Np+self.nfeats+1:-1]
        pnnX = pnn_data[:,:self.Np+self.nfeats]
        pnnY = pnn_data[:,self.Np+self.nfeats+1:-1]
        #print(X)
        if set_norm_params: # only training set, not val
            self.ham_mean = np.mean(X) if self.normalize else 0
            self.ham_var = np.var(X) if self.normalize else 1
            self.pnn_mean = np.mean(pnnX) if self.normalize else 0
            self.pnn_var = np.var(pnnX) if self.normalize else 1
            #print('mean, var = ',self.mean, self.var)
        X = X-self.ham_mean
        X = X/self.ham_var**0.5
        pnnX = pnnX-self.pnn_mean
        pnnX = pnnX/self.pnn_var**0.5
        #print(X)
        X[:,1:-1] += np.random.rand(self.time_steps*len(alphas),self.Np-2)*self.noice_level-self.noice_level/2
        pnnX[:,1:-1] += np.random.rand(self.time_steps*len(alphas),self.Np-2)*self.noice_level-self.noice_level/2
        return X,Y, pnnX, pnnY



    def train(self, figname=None):

        start_time = datetime.datetime.now()
        X, Y, pnnX, pnnY = self.__data_set(self.alpha_train, set_norm_params=True)
        X_val, Y_val, pnnX_val, pnnY_val = self.__data_set(self.alpha_val)
        print(f'\nTime making data set: {datetime.datetime.now()-start_time}')

        start_time = datetime.datetime.now()
        train_kwargs = {
                #'epochs':self.epochs, 
                'batch_size':32, 
                #'callbacks': [keras.callbacks.EarlyStopping(patience=self.patience)], 
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }
        ham_train_hists = []
        ham_val_hists = []
        for model in self.hamNNs:
            # train for minimum 100 steps
            train_result = model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
            train_hist = train_result.history['loss']
            val_hist = train_result.history['val_loss']
            # train with patience 20
            train_result = model.fit(X, Y, 
                    epochs=self.epochs, 
                    callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])], 
                    **train_kwargs)
            train_hist.extend(train_result.history['loss'])
            val_hist.extend(train_result.history['val_loss'])
            ham_train_hists.append(train_hist)
            ham_val_hists.append(val_hist)
        print(f'\nTime training ham NNs: {datetime.datetime.now()-start_time}')

        # pure NN:
        train_kwargs['validation_data'] = (pnnX_val, pnnY_val)
        pnn_train_hists = []
        pnn_val_hists = []
        for model in self.pureNNs:
            # train for minimum 100 steps
            train_result = model.fit(pnnX, pnnY, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
            train_hist = train_result.history['loss']
            val_hist = train_result.history['val_loss']
            # train with patience 20
            train_result = model.fit(pnnX, pnnY, 
                    epochs=self.epochs, 
                    callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])], 
                    **train_kwargs)
            train_hist.extend(train_result.history['loss'])
            val_hist.extend(train_result.history['val_loss'])
            pnn_train_hists.append(train_hist)
            pnn_val_hists.append(val_hist)
        print(f'\nTime training all NNs: {datetime.datetime.now()-start_time}')

        fig, axs = plt.subplots(1,2)
        for hist in ham_train_hists:
            epochs_vector = np.arange(1, len(hist)+1)
            axs[0].plot(epochs_vector, hist, 'b--', label='train_losses')
        for hist in ham_val_hists:
            epochs_vector = np.arange(1, len(hist)+1)
            axs[0].plot(epochs_vector, hist, 'r', label='val_losses')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Number of epochs')
        axs[0].set_ylabel('Loss (MSE)')
        axs[0].legend(title='hamNN')
        axs[0].grid()
        for hist in pnn_train_hists:
            epochs_vector = np.arange(1, len(hist)+1)
            axs[1].plot(epochs_vector, hist, 'b--', label='train_losses')
        for hist in pnn_val_hists:
            epochs_vector = np.arange(1, len(hist)+1)
            axs[1].plot(epochs_vector, hist, 'r', label='val_losses')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Number of epochs')
        axs[1].set_ylabel('Loss (MSE)')
        axs[1].legend(title='pureNN')
        axs[1].grid()
        if figname != None:
            plt.savefig(figname)
        if self.plot:
            plt.show()
        else:
            plt.close()


    def __call__(self, alpha=None, model_index=0):
        NN = self.hamNNs[model_index]
        if alpha == None:
            alpha = self.alpha_test[0]
        X=np.zeros((1,self.Np+self.nfeats))
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
            if self.alpha_feature:
                X[:,-1] = alpha
            X = (X-self.ham_mean)/self.ham_var**0.5
            correction = np.zeros(self.Np)
            correction[1:-1] = NN(X)[0,:]
            #correction = NN(X)[0,:].numpy()
            #u_prev[1:-1] = u_prev[1:-1] + correction # add correction to previous solution
            fem_model.u_fem = u_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
        return fem_model.u_fem

    def call_PNN(self, alpha=None, model_index=0):
        plot_steps=False
        NN = self.pureNNs[model_index]
        if alpha == None:
            alpha = self.alpha_test[0]
        X=np.zeros((1,self.Np+self.nfeats))
        f, u_ex = functions.sbmfact(alpha=alpha)[self.sol]
        u_NN = u_ex(self.tri, t=0)[1:-1]
        k=self.T/self.time_steps
        if plot_steps:
            plt.close()
        for t in range(self.time_steps):
            u_prev = u_NN # save previous solution
            X[:,1:self.Np-1] = u_prev
            X[:,0]=u_ex(x=0,t=t*k)
            X[:,self.Np-1]=u_ex(x=1,t=t*k)
            ##X[:,0]=u_ex(x=0,t=t*k+k) ## bdry_shift_test
            ##X[:,self.Np-1]=u_ex(x=1,t=t*k+k) ## bdry_shift_test
            if self.alpha_feature:
                X[:,-1] = alpha
            #if t%100==0:
            #    plt.plot(self.tri, X[0,:self.Np], label='prev')
            if plot_steps:
                if t%10==4:
                    print(f'time is {t*k}')
                    plt.plot(self.tri, X[0,:self.Np],'r', label='new')
                    plt.plot(self.tri, u_ex(x=self.tri, t=t*k),'k--', label='ex')
            ##x_prev = X.copy() ## input_diff_test
            X = (X-self.pnn_mean)/self.pnn_var**0.5
            u_NN = NN(X)
            ##u_NN = u_prev + NN(X) ## diff_test
        if plot_steps:
            plt.grid()
            #plt.legend()
            plt.show()
        result = np.zeros((self.Np))
        result[1:self.Np-1] = u_NN
        result[0] = u_ex(x=0,t=self.T)
        result[self.Np-1] = u_ex(x=1,t=self.T)
        return result


    def test(self, interpol = True, figname=None):
        start_time = datetime.datetime.now()
        alphas = self.alpha_test_interpol if interpol else self.alpha_test_extrapol
        fem_score = {}
        fem_score_tri = {}
        costa_score = {}
        costa_score_tri = {}
        pnn_score = {}
        pnn_score_tri = {}
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
            fem_score[f'{alpha}'] = fem_model.relative_L2()
            fem_score_tri[f'{alpha}'] = np.sqrt(np.mean((fem_model.u_fem-fem_model.u_ex(self.tri,self.T))**2)) / np.sqrt(np.mean(fem_model.u_ex(self.tri,self.T)**2))

            # Solve with HAM
            for m in range(self.NoM):
                fem_model.u_fem =self(alpha, model_index=m) # store in fem_model for easy use of relative_L2 and soltion functoins
                axs[i].plot(tri_fine, fem_model.solution(tri_fine), 'g', label='costa')
                costa_score[f'{alpha},{m}'] = fem_model.relative_L2()
                costa_score_tri[f'{alpha},{m}'] = np.sqrt(np.mean((fem_model.u_fem-fem_model.u_ex(self.tri,self.T))**2)) / np.sqrt(np.mean(fem_model.u_ex(self.tri,self.T)**2))

            # Solve with DNN
            for m in range(self.NoM):
                fem_model.u_fem =self.call_PNN(alpha, model_index=m) # store in fem_model for easy use of relative_L2 and soltion functoins
                axs[i].plot(tri_fine, fem_model.solution(tri_fine), 'y', label='pureNN')
                pnn_score[f'{alpha},{m}'] = fem_model.relative_L2()
                pnn_score_tri[f'{alpha},{m}'] = np.sqrt(np.mean((fem_model.u_fem-fem_model.u_ex(self.tri,self.T))**2)) / np.sqrt(np.mean(fem_model.u_ex(self.tri,self.T)**2))
            axs[i].plot(tri_fine, u_ex(tri_fine), 'k--', label='exact')
            axs[i].grid()
            axs[i].legend(title=f'sol={self.sol},a={alpha}')
        print(f'\nTime testing: {datetime.datetime.now()-start_time}')
        print('')
        print('FEM L2 errors:', fem_score)
        print('CoSTA L2 errors:', costa_score)
        print('PureNN L2 errors:', pnn_score)
        print('FEM l2 errors:', fem_score_tri)
        print('CoSTA l2 errors:', costa_score_tri)
        print('PureNN l2 errors:', pnn_score_tri)
        print('')
        if figname != None:
            plt.savefig(figname)
        if self.plot:
            plt.show()
        else:
            plt.close()
        return fem_score_tri, costa_score_tri, pnn_score_tri
