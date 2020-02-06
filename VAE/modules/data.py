import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist

def load_MNIST():

    # ---- Get data
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # ---- Normalization
    
    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype( 'float32') / 255.

    # ---- Reshape : (28,28) -> (28,28,1)

    x_train = np.expand_dims(x_train, axis=3)
    x_test  = np.expand_dims(x_test,  axis=3)
    
    print('Dataset loaded.')
    print('Resized and normalized.')
    print(f'x_train shape : {x_train.shape}\nx_test_shape  : {x_test.shape}')
    
    return (x_train,y_train),(x_test,y_test)