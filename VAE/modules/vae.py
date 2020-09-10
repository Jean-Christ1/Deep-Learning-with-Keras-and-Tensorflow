
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
# by JL Parouty (feb 2020), based on David Foster examples.

import numpy as np
import math

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from modules.callbacks      import ImagesCallback
from modules.data_generator import DataGenerator


import os, json, time, datetime



class VariationalAutoencoder():

    version = '1.28'
    
    def __init__(self, input_shape=None, encoder_layers=None, decoder_layers=None, z_dim=None, run_tag='000', verbose=0):
               
        self.name           = 'Variational AutoEncoder'
        self.input_shape    = list(input_shape)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.z_dim          = z_dim
        self.run_tag        = str(run_tag)
        self.verbose        = verbose
        self.run_directory  = f'./run/{run_tag}'
        
        # ---- Create run directories
        for d in ('','/models','/figs','/logs','/images'):
            os.makedirs(self.run_directory+d, mode=0o750, exist_ok=True)
        
        # ==== Encoder ================================================================
        
        # ---- Input layer
        encoder_input = Input(shape=self.input_shape, name='encoder_input')
        x = encoder_input
        
        # ---- Add next layers
        i=1
        for l_config in encoder_layers:
            l_type   = l_config['type']
            l_params = l_config.copy()
            l_params.pop('type')
            if l_type=='Conv2D':
                layer = Conv2D(**l_params)
            if l_type=='Dropout':
                layer = Dropout(**l_params)
            x = layer(x)
            i+=1
            
        # ---- Flatten
        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        
        # ---- mu <-> log_var
        self.mu      = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        # ---- output layer
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        # ==== Decoder ================================================================

        # ---- Input layer
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        
        # ---- First dense layer
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        # ---- Add next layers
        i=1
        for l_config in decoder_layers:
            l_type   = l_config['type']
            l_params = l_config.copy()
            l_params.pop('type')
            if l_type=='Conv2DTranspose':
                layer = Conv2DTranspose(**l_params)
            if l_type=='Dropout':
                layer = Dropout(**l_params)
            x = layer(x)
            i+=1

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)
        
        # ==== Encoder-Decoder ========================================================
        
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

        # ==== Verbosity ==============================================================

        print('Model initialized.')
        print(f'Outputs will be in  : {self.run_directory}')
        
        if verbose>0 :
            print('\n','-'*10,'Encoder','-'*50,'\n')
            self.encoder.summary()
            print('\n','-'*10,'Encoder','-'*50,'\n')
            self.decoder.summary()
            self.plot_model()
        
        
        
    def compile(self, optimizer='adam', r_loss_factor='1000'):
        self.r_loss_factor = r_loss_factor

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + kl_loss

        # See : https://github.com/tensorflow/tensorflow/issues/34944
        # See : https://github.com/tensorflow/probability/issues/519
        #
        # Uncomment :
        # tf.config.experimental_run_functions_eagerly(True)
        #
        # Works fine in versions 2.2, 2.3 but with horible perf. (7s -> 1'50s)
        #
        self.model.compile(optimizer=optimizer, 
                           loss = vae_loss,
                           metrics = [vae_r_loss, vae_kl_loss],
                           experimental_run_tf_function=False)
        print('Compiled.')
    
    
    
    def train(self, 
              x_train=None,
              x_test=None,
              data_generator=None,
              batch_size=32, 
              epochs=20,
              initial_epoch=0,
              k_size=1
             ):

        # ---- Data given or via generator
        mode_data = (data_generator is None)
        
        # ---- Size of the dataset we are going to use
        #      k_size ==1 : mean 100%
        #      Unused with data generator
        #
        if mode_data:
            n_train = int(x_train.shape[0] * k_size)
            n_test  = int(x_test.shape[0]  * k_size)
                   
        # ---- Callback : Images
        filename = self.run_directory+"/images/image-{epoch:03d}-{i:02d}.jpg"
        callbacks_images = ImagesCallback(filename, z_dim=self.z_dim, decoder=self.decoder)
        
        # ---- Callback : Checkpoint
        filename = self.run_directory+"/models/model-{epoch:03d}.h5"
        callback_chkpts = ModelCheckpoint(filename, save_freq='epoch' ,verbose=0)

        # ---- Callback : Best model
        filename = self.run_directory+"/models/best_model.h5"
        callback_bestmodel = ModelCheckpoint(filename, save_best_only=True, mode='min',monitor='val_loss',verbose=0)

        # ---- Callback tensorboard
        dirname = self.run_directory+"/logs"
        callback_tensorboard = TensorBoard(log_dir=dirname, histogram_freq=1)

        callbacks_list = [callbacks_images, callback_chkpts, callback_bestmodel, callback_tensorboard]
#         callbacks_list = [callback_chkpts, callback_bestmodel, callback_tensorboard]

        # ---- Let's go...
        start_time   = time.time()
        
        if mode_data:
            #
            # ---- With pure data (x_train) -----------------------------------------
            #                             
            self.history = self.model.fit(x_train[:n_train], x_train[:n_train],
                                          batch_size = batch_size,
                                          shuffle = True,
                                          epochs = epochs,
                                          initial_epoch = initial_epoch,
                                          callbacks = callbacks_list,
                                          validation_data = (x_test[:n_test], x_test[:n_test])
                                          )
            #
        else:
            # ---- With Data Generator ----------------------------------------------
            #
            self.history = self.model.fit(data_generator,
                                          shuffle = True,
                                          epochs = epochs,
                                          initial_epoch = initial_epoch,
                                          callbacks = callbacks_list,
                                          validation_data = (x_test, x_test)
                                         )
        
        end_time  = time.time()
        dt  = end_time-start_time
        dth = str(datetime.timedelta(seconds=int(dt)))
        self.duration = dt
        print(f'\nTrain duration : {dt:.2f} sec. - {dth:}')


        
        
    def plot_model(self):
        d=self.run_directory+'/figs'
        plot_model(self.model,   to_file=f'{d}/model.png',   show_shapes = True, show_layer_names = True, expand_nested=True)
        plot_model(self.encoder, to_file=f'{d}/encoder.png', show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=f'{d}/decoder.png', show_shapes = True, show_layer_names = True)

        
        
        
    def save(self,config='vae_config.json', model='model.h5', force=False):
        
        # ---- Check if the place is still used
        
        if os.path.isfile(self.run_directory+'/models/best_model.h5') and not force:
            print('\n*** Oops. There are already stuff in the target folder !\n')
            assert False, f'Tag directory {self.run_directory} is not empty...'
           
        # ---- Save config in json
        
        if config!=None:
            to_save  = ['input_shape', 'encoder_layers', 'decoder_layers', 'z_dim', 'run_tag', 'verbose']
            data     = { i:self.__dict__[i] for i in to_save }
            filename = self.run_directory+'/models/'+config
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)
            print(f'Config saved in     : {filename}')
        
        # ---- Save model
        
        if model!=None:
            filename = self.run_directory+'/models/'+model
            self.model.save(filename)
            print(f'Model saved in      : {filename}')

        
        
    def load_weights(self,model='model.h5'):
        filename = self.run_directory+'/models/'+model
        self.model.load_weights(filename)
        print(f'Weights loaded from : {filename}')
    
            
    @classmethod
    def load(cls, run_tag='000', config='vae_config.json', weights='model.h5'):
        # ---- Instantiate a new vae
        filename = f'./run/{run_tag}/models/{config}'
        with open(filename, 'r') as infile:
            params=json.load(infile)
            vae=cls( **params)
        # ---- weights==None, just return it
        if weights==None: return vae
        # ---- weights!=None, get weights
        vae.load_weights(weights)
        return vae
    
    @classmethod
    def about(cls):
        print('\nFIDLE 2020 - Variational AutoEncoder (VAE)')
        print('TensorFlow version   :',tf.__version__)
        print('VAE version          :', cls.version)