import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model

import tensorflow.keras.datasets.imdb as imdb




class VariationalAutoencoder():

    
    def __init__(self, input_shape, encoder_layers, decoder_layers, z_dim):
        
        self.name           = 'Variational AutoEncoder'
        self.input_shape    = input_shape
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.z_dim          = z_dim
        
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
        
        # ---- mu    /   log_var
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
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss], experimental_run_tf_function=False)
