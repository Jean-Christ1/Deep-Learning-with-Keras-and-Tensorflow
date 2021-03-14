from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

class ImagesCallback(Callback):
    
   
    def __init__(self, filename='image-{epoch:03d}-{i:02d}.jpg', 
                       x=None,
                       encoder=None, decoder=None):
        self.filename  = filename
        self.x         = x
        self.encoder   = encoder
        self.decoder   = decoder
        if len(x)>100:
            print('***Warning : The number of images is reduced to 100')
            self.x=x[:100]
        
    def on_epoch_end(self, epoch, logs={}):  
        
        # ---- Get latent points
        #
        z_new  = self.encoder.predict(self.x)
        
        # ---- Predict an image
        #
        images = self.decoder.predict(np.array(z_new))
        
        # ---- Save images
        #
        for i,image in enumerate(images):
            
            # ---- Squeeze it if monochrome : (lx,ly,1) -> (lx,ly) 
            #
            image = image.squeeze()
        
            # ---- Save it
            #
            filename = self.filename.format(epoch=epoch,i=i)
            
            if len(image.shape) == 2:
                plt.imsave(filename, image, cmap='gray_r')
            else:
                plt.imsave(filename, image)
