# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                            SamplingLayer
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# by JL Parouty (dec 2020), based on François Chollet example
#
# Thanks to François Chollet example : https://keras.io/examples/generative/vae

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display,Markdown

# Note : https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class VariationalLossLayer(keras.layers.Layer):
   
    def __init__(self, loss_weights=[3,7]):
        super().__init__()
        self.k1 = loss_weights[0]
        self.k2 = loss_weights[1]


    def call(self, inputs):
        
        # ---- Retrieve inputs
        #
        x, z_mean, z_log_var, y = inputs
        
        # ---- Compute : reconstruction loss
        #
        r_loss  = tf.reduce_mean( keras.losses.binary_crossentropy(x,y) ) * self.k1
        #
        # ---- Compute : kl_loss
        #
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -tf.reduce_mean(kl_loss) * self.k2
        
        # ---- Add loss
        #
        loss = r_loss + kl_loss
        self.add_loss(loss)
        
        # ---- Keep metrics
        #
        self.add_metric(loss,   aggregation='mean',name='loss')
        self.add_metric(r_loss, aggregation='mean',name='r_loss')
        self.add_metric(kl_loss,aggregation='mean',name='kl_loss')
        return y

    
    def get_config(self):
        return {'loss_weights':[self.k1,self.k2]}