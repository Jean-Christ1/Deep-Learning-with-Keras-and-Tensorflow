
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# Initial version by JL Parouty, feb 2020

import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist

class Loader_MNIST():
    
    version = '0.1'
    
    def __init__(self):
        pass
   
    @classmethod
    def get(normalize=True, expand=True, concatenate=True):

        # ---- Get data
        #
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('Dataset loaded.')
           
        # ---- Normalization
        #
        if normalize:
            x_train = x_train.astype('float32') / 255.
            x_test  = x_test.astype( 'float32') / 255.
            print('Normalized.')
            
        # ---- Reshape : (28,28) -> (28,28,1)
        #
        if expand:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test  = np.expand_dims(x_test,  axis=-1)
            print('Expanded.')

        # ---- Concatenate
        #
        if concatenate:
            x_data = np.concatenate([x_train, x_test], axis=0)
            y_data = np.concatenate([y_train, y_test])
            print('Concatenate.')
            print('x shape :', x_data.shape)
            print('y shape :', y_data.shape)
            return x_data,y_data
        else:
            print('x_train shape :', x_train.shape)
            print('y_train shape :', y_train.shape)
            print('x_test  shape :', x_test.shape)
            print('y_test  shape :', y_test.shape)
            return x_train,y_train,x_test,y_test