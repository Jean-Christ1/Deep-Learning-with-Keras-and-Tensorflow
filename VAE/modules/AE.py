import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# See : https://keras.io/api/models/model/
# See :https://keras.io/guides/customizing_what_happens_in_fit/
    
class AE(keras.Model):
    
    def __init__(self, encoder=None, decoder=None, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, inputs):
        z = self.encoder(inputs)
        y_pred = self.decoder(z)
        return y_pred
        
        
    def train_step(self, data):
        
        x, y = data
        
        with tf.GradientTape() as tape:
            z      = self.encoder(x)
            y_pred = self.decoder(z)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # ---- Compute gradients
        #
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # ---- Update weights
        #
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # ---- Update metrics (includes the metric that tracks the loss)
        #
        self.compiled_metrics.update_state(y, y_pred)
        
        # ---- Return a dict mapping metric names to current value
        #
        return {m.name: m.result() for m in self.metrics}
#         return {"loss":loss}


    
    def reload(self,filename):
        self.encoder = keras.models.load_model(f'{filename}-enc.h5')
        self.decoder = keras.models.load_model(f'{filename}-dec.h5')
        print('Reloaded.')
        
    def save(self,filename):
        self.encoder.save(f'{filename}-enc.h5')
        self.decoder.save(f'{filename}-dec.h5')