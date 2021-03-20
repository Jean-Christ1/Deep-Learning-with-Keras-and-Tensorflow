from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

class ImagesCallback(Callback):
    '''
    Save generated (random mode) or encoded/decoded (z mode) images on epoch end.
    params:
        encoder     : encoder
        decoder     : decoder
        x           : input images, for z mode (None)
        z_dim       : size of the latent space, for random mode (None)
        nb_images   : number of images to save
        from_z      : save images from z (False)
        from_random : save images from random (False)
        filename    : images filename
        run_dir     : output directory to save images        
    '''
    
   
    def __init__(self, encoder     = None, 
                       decoder     = None,
                       x           = None,
                       z_dim       = None,
                       nb_images   = 5,
                       from_z      = False, 
                       from_random = False,
                       filename    = 'image-{epoch:03d}-{i:02d}.jpg',
                       run_dir     = './run'):
        
        # ---- Parameters
        #
        self.x           = x[:nb_images]
        self.z_dim       = z_dim
        
        self.nb_images   = nb_images
        self.from_z      = from_z
        self.from_random = from_random

        self.filename_z       = run_dir + '/images-z/'      + filename
        self.filename_random  = run_dir + '/images-random/' + filename
        
        self.encoder     = encoder
        self.decoder     = decoder
                       
        if from_z:      os.makedirs( run_dir + '/images-z/',     mode=0o750, exist_ok=True)
        if from_random: os.makedirs( run_dir + '/images-random/', mode=0o750, exist_ok=True)
        
    
    
    def save_images(self, images, filename, epoch):
        '''Save images as <filename>'''
        
        for i,image in enumerate(images):
            
            image = image.squeeze()  # Squeeze it if monochrome : (lx,ly,1) -> (lx,ly) 
        
            filenamei = filename.format(epoch=epoch,i=i)
            
            if len(image.shape) == 2:
                plt.imsave(filenamei, image, cmap='gray_r')
            else:
                plt.imsave(filenamei, image)

    
    
    def on_epoch_end(self, epoch, logs={}):
        '''Called at the end of each epoch'''
        
        if self.from_random:
            z      = np.random.normal( size=(self.nb_images,self.z_dim) )
            images = self.decoder.predict(z)
            self.save_images(images, self.filename_random, epoch)
            
        if self.from_z:
            z_mean, z_var, z  = self.encoder.predict(self.x)
            images            = self.decoder.predict(z)
            self.save_images(images, self.filename_z, epoch)


    def get_images(self, epochs=None, from_z=True,from_random=True):
        '''Read and return saved images. epochs is a range'''
        if epochs is None : return
        images_z = []
        images_r = []
        for epoch in list(epochs):
            for i in range(self.nb_images):
                if from_z:
                    f = self.filename_z.format(epoch=epoch,i=i)
                    images_z.append( io.imread(f) )
                if from_random:
                    f = self.filename_random.format(epoch=epoch,i=i)
                    images_r.append( io.imread(f) )
        return images_z, images_r
            

    
                
class BestModelCallback(Callback):

    def __init__(self, filename= './run_dir/best-model', verbose=0 ):
        self.filename = filename
        self.verbose  = verbose
        self.loss     = np.Inf
        os.makedirs( os.path.dirname(filename), mode=0o750, exist_ok=True)
                
    def on_train_begin(self, logs=None):
        self.loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if current < self.loss:
            self.loss = current
            self.model.save(self.filename)
            if self.verbose>0: print(f'Saved - loss={current:.6f}')
