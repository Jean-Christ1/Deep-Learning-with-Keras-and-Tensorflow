
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
    def about(cls):
        print('\nFIDLE 2020 - Very basic MNIST dataset loader)')
        print('TensorFlow version   :',tf.__version__)
        print('Loader version       :', cls.version)
    
    @classmethod
    def load(normalize=True, expand=True, verbose=1):

        # ---- Get data

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if verbose>0: print('Dataset loaded.')


        # ---- Normalization

        if normalize:
            x_train = x_train.astype('float32') / 255.
            x_test  = x_test.astype( 'float32') / 255.
            if verbose>0: print('Normalized.')
            
        # ---- Reshape : (28,28) -> (28,28,1)

        if expand:
            x_train = np.expand_dims(x_train, axis=3)
            x_test  = np.expand_dims(x_test,  axis=3)
            if verbose>0: print(f'Reshaped to {x_train.shape}')

        return (x_train,y_train),(x_test,y_test)