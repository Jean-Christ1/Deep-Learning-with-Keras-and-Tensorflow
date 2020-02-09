from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os

class ImagesCallback(Callback):
    
    def __init__(self, initial_epoch=0, image_periodicity=1, vae=None):
        self.epoch             = initial_epoch
        self.image_periodicity = image_periodicity
        self.vae               = vae
        self.images_dir        = vae.run_directory+'/images'
        batch_per_epochs       = int(vae.n_train / vae.batch_size)
        self.batch_periodicity = batch_per_epochs*image_periodicity
        
        
    def on_train_batch_end(self, batch, logs={}):  
        
        if batch % self.batch_periodicity == 0:
            # ---- Get a random latent point
            z_new   = np.random.normal(size = (1,self.vae.z_dim))
            # ---- Predict an image
            image = self.vae.decoder.predict(np.array(z_new))[0]
            # ---- Squeeze it if monochrome : (lx,ly,1) -> (lx,ly) 
            image = image.squeeze()
            # ---- Save it
            filename=f'{self.images_dir}/img_{self.epoch:05d}_{batch:06d}.jpg'
            if len(image.shape) == 2:
                plt.imsave(filename, image, cmap='gray_r')
            else:
                plt.imsave(filename, image)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1
