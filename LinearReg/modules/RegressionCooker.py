
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|  Regression cooker
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# Initial version by JL Parouty, feb 2020

import numpy as np
import math
import random
import datetime, time, sys

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,HTML

sys.path.append('..')
import fidle.pwk as pwk

class RegressionCooker():
    
    pwk     = None
    version = '0.1'
    
    def __init__(self, pwk):
        self.pwk = pwk
        pwk.subtitle('FIDLE 2020 - Regression Cooker')
        print('Version      :', self.version)
        print('Run time     : {}'.format(time.strftime("%A %d %B %Y, %H:%M:%S")))
        

    @classmethod
    def about(cls):
        print('\nFIDLE 2020 - Regression Cooker)')
        print('Version       :', cls.version)
    
    
    @classmethod
    def vector_infos(cls,name,V):
        """
        Show some nice infos about a vector
        args:
            name : vector name
            V    : vector
        """
        m=V.mean(axis=0).item()
        s=V.std(axis=0).item()
        print("{:16} :      mean={:8.3f}  std={:8.3f}    min={:8.3f}    max={:8.3f}".format(name,m,s,V.min(),V.max()))
    
    
    def get_dataset(self,n):
        """
        Return a dataset of n observation
        args:
            n       : dataset size
        return:
            X,Y : with X shapes = (n,1) Y shape = (n,)
        """
        
        xob_min   = 0       # x min and max
        xob_max   = 10

        a_min     = -30     # a min and max
        a_max     =  30
        b_min     = -10     # b min and max
        b_max     =  10

        noise_min =  10     # noise min and max
        noise_max =  50

        a0    = random.randint(a_min,a_max)       
        b0    = random.randint(b_min,b_max)       
        noise = random.randint(noise_min,noise_max)

        # ---- Construction du jeu d'apprentissage ---------------
        #      X,Y              : données brutes

        X = np.random.uniform(xob_min,xob_max,(n,1))
        N = noise * np.random.normal(0,1,(n,1))
        Y = a0*X + b0 + N

        return X,Y
    

    
    def plot_dataset(self,X,Y,title='Dataset :',width=12,height=6):
        """
        Plot dataset X,Y
        args:
            X : Observations
            Y : Values
        """
        nb_viz = min(1000,len(X))
        display(Markdown(f'### {title}'))
        print(f"X shape : {X.shape}  Y shape : {Y.shape}  plot : {nb_viz} points")
        plt.figure(figsize=(width, height))
        plt.plot(X[:nb_viz], Y[:nb_viz], '.')
        self.pwk.save_fig('01-dataset')
        plt.show()
        self.vector_infos('X',X)
        self.vector_infos('Y',Y)

        
    def __plot_theta(self, i, theta,x_min,x_max, loss,gradient,alpha):
        Xd = np.array([[x_min], [x_max]])
        Yd = Xd * theta.item(1) + theta.item(0)
        plt.plot(Xd, Yd, color=(1.,0.4,0.3,alpha))
        if i<0:
            print( "    #i   Loss       Gradient         Theta")
        else:
            print("  {:3d}  {:+7.3f}  {:+7.3f} {:+7.3f}  {:+7.3f} {:+7.3f}".format(i,loss,gradient.item(0),
                                                                                   gradient.item(1),theta.item(0),
                                                                                   theta.item(1)))

            
    def __plot_XY(self, X,Y,width=12,height=6):
        nb_viz = min(1000,len(X))
        plt.figure(figsize=(width, height))
        plt.plot(X[:nb_viz], Y[:nb_viz], '.') 
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        
    def __plot_loss(self,loss, width=8,height=4):
        plt.figure(figsize=(width, height))
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.ylim(0, 20)
        plt.plot(range(len(loss)), loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        
    def basic_descent(self, X, Y, epochs=200, eta=0.01,width=12,height=6):
        """
        Performs a gradient descent where the gradient is updated at the end
        of each iteration for all observations.
        args:
            X,Y          : Observations
            epochs       : Number of epochs (200)
            eta          : learning rate
            width,height : graphic size
        return:
            theta        : theta
        """

        display(Markdown(f'### Basic gradient descent :'))

        display(Markdown(f'**With :**  '))
        print('with :')
        print(f'    epochs = {epochs}')
        print(f'    eta    = {eta}')

        display(Markdown(f'**epochs :**  '))
        x_min = X.min()
        x_max = X.max()
        y_min = Y.min()
        y_max = Y.max()
        n     = len(X)
        
        # ---- Initialization

        theta = np.array([[y_min],[0]])
        X_b = np.c_[np.ones((n, 1)), X]

        # ---- Visualization
        
        self.__plot_XY(X,Y,width,height)
        self.__plot_theta( -1, theta,x_min,x_max, None,None,0.1)
        
        # ---- Training
        
        loss=[]
        for i in range(epochs+1):

            gradient = (2/n) * X_b.T @ ( X_b @ theta - Y)
            mse = ((X_b @ theta - Y)**2).mean(axis=None)

            theta = theta - eta * gradient
            
            loss.append(mse)
            if (i % (epochs/10))==0:
                self.__plot_theta( i, theta,x_min,x_max, mse,gradient,i/epochs)

        # ---- Visualization

        pwk.subtitle('Visualization :')
        self.pwk.save_fig('02-basic_descent')
        plt.show()

        pwk.subtitle('Loss :')
        self.__plot_loss(loss)
        self.pwk.save_fig('03-basic_descent_loss')
        plt.show()
        
        return theta
    
    
    def minibatch_descent(self, X, Y, epochs=200, batchs=5, batch_size=10, eta=0.01,width=12,height=6):
        """
        Performs a gradient descent where the gradient is updated at the end
        of each iteration for all observations.
        args:
            X,Y          : Observations
            epochs       : Number of epochs (200)
            eta          : learning rate
            width,height : graphic size
        return:
            theta        : theta
        """

        display(Markdown(f'### Mini batch gradient descent :'))

        display(Markdown(f'**With :**  '))
        print('with :')
        print(f'    epochs     = {epochs}')
        print(f'    batchs     = {batchs}')
        print(f'    batch size = {batch_size}')
        print(f'    eta        = {eta}')

        display(Markdown(f'**epochs :**  '))
        x_min = X.min()
        x_max = X.max()
        y_min = Y.min()
        y_max = Y.max()
        n     = len(X)
        
        # ---- Initialization

        theta = np.array([[y_min],[0]])
        X_b = np.c_[np.ones((n, 1)), X]

        # ---- Visualization
        
        self.__plot_XY(X,Y,width,height)
        self.__plot_theta( -1, theta,x_min,x_max, None,None,0.1)

        # ---- Training

        def learning_schedule(t):
            return 1 / (t + 100)

        loss=[]
        for epoch in range(epochs):
            for i in range(batchs):

                random_index = np.random.randint(n-batch_size)

                xi = X_b[random_index:random_index+batch_size]
                yi = Y[random_index:random_index+batch_size]

                mse = ((xi @ theta - yi)**2).mean(axis=None)
                gradient = 2 * xi.T.dot(xi.dot(theta) - yi)

                eta = learning_schedule(epoch*150)
                theta = theta - eta * gradient

            loss.append(mse)
            self.__plot_theta( epoch, theta,x_min,x_max, mse,gradient,epoch/epochs)
#             draw_theta(epoch,mse,gradients, theta,0.1+epoch/(n_epochs+1))

#         draw_theta(epoch,mse,gradients,theta,1)
        
        # ---- Visualization

        pwk.subtitle('Visualization :')
        self.pwk.save_fig('04-minibatch_descent')
        plt.show()

        pwk.subtitle('Loss :')
        self.__plot_loss(loss)
        self.pwk.save_fig('05-minibatch_descent_loss')
        plt.show()
        
        
        return theta
