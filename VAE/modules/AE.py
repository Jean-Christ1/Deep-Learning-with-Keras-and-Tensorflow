import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

   
    
    
class AE(keras.Model):
    
    def __init__(self, encoder=None, decoder=None, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        
    def train_step(self, data):
        
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            z                    = self.encoder(data)
            reconstruction       = self.decoder(z)
            reconstruction_loss  = tf.reduce_mean( keras.losses.binary_crossentropy(data, reconstruction) )
            reconstruction_loss *= 28*28
            
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "loss": reconstruction_loss
        }
    
    
    def reload(self,filename):
        self.encoder = keras.models.load_model(f'{filename}-enc.h5', custom_objects={'Sampling': Sampling})
        self.decoder = keras.models.load_model(f'{filename}-dec.h5')
        
    def save(self,filename):
        self.encoder.save(f'{filename}-enc.h5')
        self.decoder.save(f'{filename}-dec.h5')