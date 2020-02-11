from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

class ImagesCallback(Callback):
    
    def __init__(self, filename= 'image-{epoch:03d}.jpg', z_dim=0, decoder=None):
        self.filename = filename
        self.z_dim    = z_dim
        self.decoder  = decoder
        
        
    def on_epoch_end(self, epoch, logs={}):  
        
        # ---- Get a random latent point
        
        z_new   = np.random.normal(size = (1,self.z_dim))
        
        # ---- Predict an image
        
        image = self.decoder.predict(np.array(z_new))[0]
        
        # ---- Squeeze it if monochrome : (lx,ly,1) -> (lx,ly) 
        
        image = image.squeeze()
        
        # ---- Save it

        filename = self.filename.format(epoch=epoch)
        if len(image.shape) == 2:
            plt.imsave(filename, image, cmap='gray_r')
        else:
            plt.imsave(filename, image)
