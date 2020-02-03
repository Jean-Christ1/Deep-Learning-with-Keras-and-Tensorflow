import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import tensorflow.keras.datasets.imdb as imdb

import modules.callbacks
import os



class VariationalAutoencoder():

    
    def __init__(self, input_shape=None, encoder_layers=None, decoder_layers=None, z_dim=None, run_tag='default', verbose=0):
        
        self.name           = 'Variational AutoEncoder'
        self.input_shape    = input_shape
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.z_dim          = z_dim
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
        for params in encoder_layers:
            t=params['type']
            params.pop('type')
            if t=='Conv2D':
                layer = Conv2D(**params, name=f"Layer_{i}")
            if t=='Dropout':
                layer = Dropout(**params)
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
        for params in decoder_layers:
            t=params['type']
            params.pop('type')
            if t=='Conv2DT':
                layer = Conv2DTranspose(**params, name=f"Layer_{i}")
            if t=='Dropout':
                layer = Dropout(**params)
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
        print('Outputs will be in : ',self.run_directory)
        
        if verbose>0 :
            print('\n','-'*10,'Encoder','-'*50,'\n')
            self.encoder.summary()
            print('\n','-'*10,'Encoder','-'*50,'\n')
            self.decoder.summary()
            self.plot_model()
        
        
        
    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate
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

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss = vae_loss,
                           metrics = [vae_r_loss, vae_kl_loss], 
                           experimental_run_tf_function=False)
    
    
    def train(self, 
              x_train,x_test,
              batch_size=32, epochs=200, 
              batch_periodicity=100, 
              initial_epoch=0,
              dataset_size=1,
              lr_decay=1):

        # ---- Dataset size
        n_train = int(x_train.shape[0] * dataset_size)
        n_test  = int(x_test.shape[0]  * dataset_size)

        # ---- Callbacks
        images_callback = modules.callbacks.ImagesCallback(initial_epoch, batch_periodicity, self)
        
#         lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)
        
        filename1 = self.run_directory+"/models/model-{epoch:03d}-{loss:.2f}.h5"
        batch_per_epoch = int(len(x_train)/batch_size)
        checkpoint1 = ModelCheckpoint(filename1, save_freq=batch_per_epoch*5,verbose=0)

        filename2 = self.run_directory+"/models/best_model.h5"
        checkpoint2 = ModelCheckpoint(filename2, save_best_only=True, mode='min',monitor='val_loss',verbose=0)

        callbacks_list = [checkpoint1, checkpoint2, images_callback]

        self.model.fit(x_train[:n_train], x_train[:n_train],
                       batch_size = batch_size,
                       shuffle = True,
                       epochs = epochs,
                       initial_epoch = initial_epoch,
                       callbacks = callbacks_list,
                       validation_data = (x_test[:n_test], x_test[:n_test])
                        )
        
        
    def plot_model(self):
        d=self.run_directory+'/figs'
        plot_model(self.model,   to_file=f'{d}/model.png',   show_shapes = True, show_layer_names = True, expand_nested=True)
        plot_model(self.encoder, to_file=f'{d}/encoder.png', show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=f'{d}/decoder.png', show_shapes = True, show_layer_names = True)
