from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

class ImagesCallback(Callback):
    '''
    Save generated or encoded/decoded images on epoch end.
    params:
        filename : images filename
        x : input images for encoded/decoded or None for generated mode
        nb_images : number of images (if generated mode)
        z_dim : size of the latent space (if generated mode)
        encoder : encoder
        decoder : decoder
    '''
    
   
    def __init__(self, filename='image-{epoch:03d}-{i:02d}.jpg', 
                       x=None, nb_images=5, z_dim=None,
                       encoder=None, decoder=None):
        self.filename  = filename
        self.x         = x
        self.nb_images = nb_images
        self.z_dim     = z_dim
        self.encoder   = encoder
        self.decoder   = decoder
        if x is not None:
            if len(x)>100:
                print('***Warning : The number of images is reduced to 100')
                self.x=x[:100]
        
    def on_epoch_end(self, epoch, logs={}):  
               
        # ---- Get latent points
        #
        if self.x is None:
            z = np.random.normal( size=(self.nb_images,self.z_dim) )
        else:
            z_mean, z_var, z  = self.encoder.predict(self.x)
        
        # ---- Predict an image
        #
        images = self.decoder.predict(z)
        
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


    
                
class BestModelCallback(Callback):

    def __init__(self, filename= 'best-model' ):
        self.filename = filename
        self.loss     = np.Inf
        
    def on_train_begin(self, logs=None):
        self.loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if current < self.loss:
            self.loss = current
            self.model.save(self.filename)
            print(f'Saved - loss={current:.6f}')
