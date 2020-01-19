#!/usr/bin/env python
# coding: utf-8

# German Traffic Sign Recognition Benchmark (GTSRB)
# =================================================
# ---
# Introduction au Deep Learning  (IDLE) - S. Arias, E. Maldonado, JL. Parouty - CNRS/SARI/DEVLOG - 2020  
# 
# ## Episode 5 : Full Convolutions
# 
# Our main steps:
#  - Try n models with n datasets
#  - Save a Pandas/h5 report
#  - Can be run in :
#     - Notebook mode
#     - Batch mode 
#     - Tensorboard follow up
#     
# To export a notebook as a script :  
# ```jupyter nbconvert --to script <notebook>```
# 
# To run a notebook :  
# ```jupyter nbconvert --to notebook --execute <notebook>```
# 
# ## 1/ Import

# In[1]:


import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import h5py
import os,time

from IPython.display import display

VERSION='1.2'


# ## 2/ Init and start

# In[2]:


print('\nFull Convolutions Notebook')
print('  Version            : {}'.format(VERSION))
print('  Run time           : {}'.format(time.strftime("%A %-d %B %Y, %H:%M:%S")))
print('  TensorFlow version :',tf.__version__)
print('  Keras version      :',tf.keras.__version__)


# ## 3/ Dataset loading

# In[3]:


def read_dataset(name):
    '''Reads h5 dataset from ./data

    Arguments:  dataset name, without .h5
    Returns:    x_train,y_train,x_test,y_test data'''
    # ---- Read dataset
    filename='./data/'+name+'.h5'
    with  h5py.File(filename) as f:
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test  = f['x_test'][:]
        y_test  = f['y_test'][:]

    return x_train,y_train,x_test,y_test


# ## 4/ Models collection

# In[4]:



# A basic model
#
def get_model_v1(lx,ly,lz):
    
    model = keras.models.Sequential()
    
    model.add( keras.layers.Conv2D(96, (3,3), activation='relu', input_shape=(lx,ly,lz)))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(192, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Flatten()) 
    model.add( keras.layers.Dense(1500, activation='relu'))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Dense(43, activation='softmax'))
    return model
    
# A more sophisticated model
#
def get_model_v2(lx,ly,lz):
    model = keras.models.Sequential()

    model.add( keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(lx,ly,lz), activation='relu'))
    model.add( keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add( keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add( keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Flatten())
    model.add( keras.layers.Dense(512, activation='relu'))
    model.add( keras.layers.Dropout(0.5))
    model.add( keras.layers.Dense(43, activation='softmax'))
    return model

# My sphisticated model, but small and fast
#
def get_model_v3(lx,ly,lz):
    model = keras.models.Sequential()
    model.add( keras.layers.Conv2D(32, (3,3),   activation='relu', input_shape=(lx,ly,lz)))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Flatten()) 
    model.add( keras.layers.Dense(1152, activation='relu'))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Dense(43, activation='softmax'))
    return model


# ## 5/ Multiple datasets, multiple models ;-)

# In[5]:


def multi_run(datasets, models, batch_size=64, epochs=16):

    # ---- Columns of report
    #
    report={}
    report['Dataset']=[]
    report['Size']   =[]
    for m in models:
        report[m+' Accuracy'] = []
        report[m+' Duration'] = []

    # ---- Let's go
    #
    for d_name in datasets:
        print("\nDataset : ",d_name)

        # ---- Read dataset
        x_train,y_train,x_test,y_test = read_dataset(d_name)
        d_size=os.path.getsize('./data/'+d_name+'.h5')/(1024*1024)
        report['Dataset'].append(d_name)
        report['Size'].append(d_size)
        
        # ---- Get the shape
        (n,lx,ly,lz) = x_train.shape

        # ---- For each model
        for m_name,m_function in models.items():
            print("    Run model {}  : ".format(m_name), end='')
            # ---- get model
            try:
                model=m_function(lx,ly,lz)
                # ---- Compile it
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                # ---- Callbacks tensorboard
                log_dir = "./run/logs/tb_{}_{}".format(d_name,m_name)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                # ---- Callbacks bestmodel
                save_dir = "./run/models/model_{}_{}.h5".format(d_name,m_name)
                bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, monitor='accuracy', save_best_only=True)
                # ---- Train
                start_time = time.time()
                history = model.fit(  x_train[:1000], y_train[:1000],
                                    batch_size      = batch_size,
                                    epochs          = epochs,
                                    verbose         = 0,
                                    validation_data = (x_test, y_test),
                                    callbacks       = [tensorboard_callback, bestmodel_callback])
                # ---- Result
                end_time = time.time()
                duration = end_time-start_time
                accuracy = max(history.history["val_accuracy"])*100
                #
                report[m_name+' Accuracy'].append(accuracy)
                report[m_name+' Duration'].append(duration)
                print("Accuracy={:.2f} and Duration={:.2f})".format(accuracy,duration))
            except:
                report[m_name+' Accuracy'].append('-')
                report[m_name+' Duration'].append('-')
                print('-')
    return report


# ## 6/ Run
# ### 6.1/ Clean

# In[6]:


get_ipython().run_cell_magic('bash', '', '\n/bin/rm -r ./run/logs   2>/dev/null\n/bin/rm -r ./run/models 2>/dev/null\n/bin/mkdir -p -m 755 ./run/logs\n/bin/mkdir -p -m 755 ./run/models\necho -e "\\nReset directories : ./run/logs and ./run/models ."')


# ### 6.2 Start Tensorboard

# In[22]:


get_ipython().run_cell_magic('bash', '', 'tensorboard_start --logdir ./run/logs')


# ### 6.3/ run and save report

# In[24]:


get_ipython().run_cell_magic('time', '', '\nprint(\'\\n---- Run\',\'-\'*50)\n\n# ---- Datasets and models list\n\n# For tests\ndatasets = [\'set-24x24-L\', \'set-24x24-RGB\']\nmodels   = {\'v1\':get_model_v1, \'v3\':get_model_v3}\n\n# The real one\n# datasets = [\'set-24x24-L\', \'set-24x24-RGB\', \'set-48x48-L\', \'set-48x48-RGB\', \'set-24x24-L-LHE\', \'set-24x24-RGB-HE\', \'set-48x48-L-LHE\', \'set-48x48-RGB-HE\']\n# models   = {\'v1\':get_model_v1, \'v2\':get_model_v2, \'v3\':get_model_v3}\n\n# ---- Report name\n\nreport_name=\'./run/report-{}.h5\'.format(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))\n\n# ---- Run\n\nout    = multi_run(datasets, models, batch_size=64, epochs=2)\n\n# ---- Save report\n\noutput = pd.DataFrame (out)\nparams = pd.DataFrame( {\'datasets\':datasets, \'models\':list(models.keys())} )\n\noutput.to_hdf(report_name, \'output\')\nparams.to_hdf(report_name, \'params\')\n\nprint(\'\\nReport saved as \',report_name)\nprint(\'-\'*59)')


# ### 6.4/ Stop Tensorboard

# In[23]:


get_ipython().run_cell_magic('bash', '', 'tensorboard_stop')


# ## 7/ That's all folks..

# In[21]:


print('\n{}'.format(time.strftime("%A %-d %B %Y, %H:%M:%S")))
print("The work is done.\n")


# In[ ]:




