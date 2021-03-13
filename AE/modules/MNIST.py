
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
# 2.0 version by JL Parouty, feb 2021

import h5py
import os
import numpy as np
from hashlib import blake2b
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist

class MNIST():
    
    version = '0.1'
    
    def __init__(self):
        pass
   
    @classmethod
    def get_origine(cls, normalize=True, expand=True, concatenate=True):
        """
        Return original MNIST dataset
        args:
            normalize   : Normalize dataset or not (True)
            expand      : Reshape images as (28,28,1) instead (28,28) (True)
            concatenate : Concatenate train and test sets (True)
        returns:
            x_data,y_data                   if concatenate is False
            x_train,y_train,x_test,y_test   if concatenate is True
        """

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
            print('Reshaped.')

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
        
        
    @classmethod
    def save_prepared_dataset(cls, clean_data, noisy_data, class_data, filename='./data/mnist-noisy.h5'):
        """
        Save a prepared dataset in a h5 file
        args:
            clean_data, noisy_data, class_data : clean, noisy and class dataset
            filename                      : filename
        return:
            None
        """
        path=os.path.dirname(filename)
        os.makedirs(path, mode=0o750, exist_ok=True)

        with h5py.File(filename, "w") as f:
            f.create_dataset("clean_data", data=clean_data)
            f.create_dataset("noisy_data", data=noisy_data)
            f.create_dataset("class_data", data=class_data)
        print('Saved.')
        print('clean_data shape is : ',clean_data.shape)
        print('noisy_data shape is : ',noisy_data.shape)
        print('class_data shape is : ',class_data.shape)
            
            
    @classmethod    
    def reload_prepared_dataset(cls, scale=1., train_prop=0.8, shuffle=True, seed=None, filename='./data/mnist-noisy.h5'):
        """
        Reload a saved dataset
        args:
            scale      : Scale of dataset to use. 1. mean 100% (1.)
            train_prop : Ratio of train/test
            shuffle    : Shuffle data if True
            seed       : Random seed value (no seed if None) (None)
            filename   : filename of the prepared dataset
        returns:
            clean_train,clean_test, noisy_train,noisy_test, class_train,class_test
        """
        # ---- Load saved dataset
        #
        with  h5py.File(filename,'r') as f:
            clean_data  = f['clean_data'][:]
            noisy_data  = f['noisy_data'][:]
            class_data  = f['class_data'][:]
        print('Loaded.')
        
        # ---- Rescale
        #
        n = int(scale*len(clean_data))
        clean_data, noisy_data, class_data = clean_data[:n], noisy_data[:n], class_data[:n]
        print(f'rescaled ({scale}).') 
        
        # ---- Seed
        #
        if seed is not None:
            np.random.seed(seed)
            print(f'Seeded ({seed})')
        
        # ---- Shuffle
        #
        if shuffle:
            p = np.random.permutation(len(clean_data))
            clean_data, noisy_data, class_data = clean_data[p], noisy_data[p], class_data[p]
            print('Shuffled.')
        
        # ---- Split
        #
        n=int(len(clean_data)*train_prop)
        clean_train, clean_test = clean_data[:n], clean_data[n:]
        noisy_train, noisy_test = noisy_data[:n], noisy_data[n:]
        class_train, class_test = class_data[:n], class_data[n:]
        print(f'splited ({train_prop}).') 

        # ---- Hash
        #
        h = blake2b(digest_size=10)
        for a in [clean_train,clean_test, noisy_train,noisy_test, class_train,class_test]:
            h.update(a)
        
        print('clean_train shape is : ', clean_train.shape)
        print('clean_test  shape is : ', clean_test.shape)
        print('noisy_train shape is : ', noisy_train.shape)
        print('noisy_test  shape is : ', noisy_test.shape)
        print('class_train shape is : ', class_train.shape)
        print('class_test  shape is : ', class_test.shape)
        print('Blake2b digest is    : ', h.hexdigest())
        return  clean_train,clean_test, noisy_train,noisy_test, class_train,class_test