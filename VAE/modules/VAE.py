import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Note : https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

class Sampling(keras.layers.Layer):
    '''
    A layer that receive (z_mean, z_var) '''
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

class VAE(keras.Model):
    
    def __init__(self, encoder=None, decoder=None, r_loss_factor=1., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor
        print('r_loss_factor=',self.r_loss_factor)

        
    def train_step(self, data):
        
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction       = self.decoder(z)
            reconstruction_loss  = tf.reduce_mean( keras.losses.binary_crossentropy(data, reconstruction) )
            reconstruction_loss *= 28*28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = self.r_loss_factor*reconstruction_loss + (1-self.r_loss_factor)*kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "loss":                total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss":             kl_loss,
        }
    
    
    def reload(self,filename):
        self.encoder = keras.models.load_model(f'{filename}-enc.h5', custom_objects={'Sampling': Sampling})
        self.decoder = keras.models.load_model(f'{filename}-dec.h5')
        
    def save(self,filename):
        self.encoder.save(f'{filename}-enc.h5')
        self.decoder.save(f'{filename}-dec.h5')