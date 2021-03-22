
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/       DataGenerator
#    |_|   |_|\__,_|_|\___|       for clustered CelebA sataset
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# Initial version by JL Parouty, feb 2020


import numpy as np
import pandas as pd
import math
import os,glob
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from IPython.display import display,Markdown

class DataGenerator(Sequence):

    version = '0.4.1'
    
    def __init__(self, clusters_dir='./data', batch_size=32, debug=False, scale=1):
        '''
        Instanciation of the data generator
        args:
            cluster_dir : directory of the clusters files
            batch_size  : batch size (32)
            debug       : debug mode (False)
            scale       : scale of dataset to use. 1. mean 100% (1.)
        '''
        if debug : self.about()
        #
        # ---- Get the list of clusters
        #      
        clusters_name = [ os.path.splitext(f)[0] for f in glob.glob( f'{clusters_dir}/*.npy') ]
        clusters_size = len(clusters_name)
        #
        # ---- Read each cluster description
        #      because we need the full dataset size
        #
        dataset_size  = 0
        for c in clusters_name:
            df = pd.read_csv(c+'.csv', header=0)
            dataset_size+=len(df.index)
        #
        # ---- If we only want to use a part of the dataset...
        #
        dataset_size = int(dataset_size * scale)
        #
        if debug: 
            print(f'\nClusters nb  : {len(clusters_name)} files')
            print(f'Dataset size : {dataset_size}')
            print(f'Batch size   : {batch_size}')

        #
        # ---- Remember all of that
        #
        self.clusters_dir  = clusters_dir
        self.batch_size    = batch_size
        self.clusters_name = clusters_name
        self.clusters_size = clusters_size
        self.dataset_size  = dataset_size
        self.debug         = debug
        #
        # ---- Read a first cluster
        #
        self.rewind()
    
    
    def rewind(self):
        self.cluster_i = self.clusters_size
        self.read_next_cluster()

        
    def __len__(self):
        return math.floor(self.dataset_size / self.batch_size)

    
    def __getitem__(self, idx):
        #
        # ---- Get the next item index
        #
        i=self.data_i
        #
        # ---- Get a batch
        #
        batch = self.data[i:i+self.batch_size]
        #
        # ---- Cluster is large enough
        #
        if len(batch) == self.batch_size:
            self.data_i += self.batch_size
            if self.debug: print(f'({len(batch)}) ',end='')
            return batch,batch
        #
        # ---- Not enough...
        #
        if self.debug: print(f'({len(batch)}..) ',end='')
        #
        self.read_next_cluster()
        batch2 = self.data[ 0:self.batch_size-len(batch) ]
        self.data_i = self.batch_size-len(batch)
        batch  = np.concatenate( (batch,batch2) )
        #
        if self.debug: print(f'(..{len(batch2)}) ',end='')
        return batch, batch
    
    
    def on_epoch_end(self):
        self.rewind()
    
    
    def read_next_cluster(self):
        #
        # ---- Get the next cluster name
        #      If we have reached the end of the list, we mix and
        #      start again from the beginning. 
        #
        i = self.cluster_i + 1
        if i >= self.clusters_size:
            np.random.shuffle(self.clusters_name)
            i = 0
            if self.debug : print(f'\n[shuffle!]')
        #
        # ---- Read it (images still normalized)
        #
        data = np.load( self.clusters_name[i]+'.npy', mmap_mode='r' )
        #
        # ---- Remember all of that
        #
        self.data      = data
        self.data_i    = 0
        self.cluster_i = i
        #
        if self.debug: print(f'\n[Load {self.cluster_i:02d},s={len(self.data):3d}] ',end='')
          
        
    @classmethod
    def about(cls):
        display(Markdown('<br>**FIDLE 2020 - DataGenerator**'))
        print('Version              :', cls.version)
        print('TensorFlow version   :', tf.__version__)
        print('Keras version        :', tf.keras.__version__)
