
# ==================================================================
#  ____                 _   _           _  __        __         _
# |  _ \ _ __ __ _  ___| |_(_) ___ __ _| | \ \      / /__  _ __| | __
# | |_) | '__/ _` |/ __| __| |/ __/ _` | |  \ \ /\ / / _ \| '__| |/ /
# |  __/| | | (_| | (__| |_| | (_| (_| | |   \ V  V / (_) | |  |   <
# |_|   |_|  \__,_|\___|\__|_|\___\__,_|_|    \_/\_/ \___/|_|  |_|\_\
#                                                        module pwk                                   
# ==================================================================
# A simple module to host some common functions for practical work
# pjluc 2020

import os
import glob
from datetime import datetime
import itertools
import datetime, time

import math
import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

VERSION='0.1.7'


# -------------------------------------------------------------
# init_all
# -------------------------------------------------------------
#
def init(mplstyle='fidle/talk.mplstyle'):
    global VERSION
    # ---- matplotlib
    matplotlib.style.use(mplstyle)
    # ---- Hello world
#     now = datetime.datetime.now()
    print('IDLE 2020 - Practical Work Module')
    print('  Version            :', VERSION)
    print('  Run time           : {}'.format(time.strftime("%A %-d %B %Y, %H:%M:%S")))
    print('  Matplotlib style   :', mplstyle)
    print('  TensorFlow version :',tf.__version__)
    print('  Keras version      :',tf.keras.__version__)
          
# -------------------------------------------------------------
# Folder cooking
# -------------------------------------------------------------
#
def tag_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

def mkdir(path):
    os.makedirs(path, mode=0o750, exist_ok=True)
      
def get_directory_size(path):
    """
    Return the directory size, but only 1 level
    args:
        path : directory path
    return:
        size in Mo
    """
    size=0
    for f in os.listdir(path):
        if os.path.isfile(path+'/'+f):
            size+=os.path.getsize(path+'/'+f)
    return size/(1024*1024)

# -------------------------------------------------------------
# shuffle_dataset
# -------------------------------------------------------------
#
def shuffle_np_dataset(x, y):
    assert (len(x) == len(y)), "x and y must have same size"
    p = np.random.permutation(len(x))
    return x[p], y[p]


def update_progress(what,i,imax):
    bar_length = min(40,imax)
    if (i%int(imax/bar_length))!=0 and i<imax:
        return
    progress  = float(i/imax)
    block     = int(round(bar_length * progress))
    endofline = '\r' if progress<1 else '\n'
    text = "{:16s} [{}] {:>5.1f}% of {}".format( what, "#"*block+"-"*(bar_length-block), progress*100, imax)
    print(text, end=endofline)


# -------------------------------------------------------------
# show_images
# -------------------------------------------------------------
#
def plot_images(x,y, indices, columns=12, x_size=1, y_size=1, colorbar=False, y_pred=None, cm='binary'):
    """
    Show some images in a grid, with legends
    args:
        X: images - Shapes must be (-1 lx,ly,1) or (-1 lx,ly,3)
        y: real classes
        indices: indices of images to show
        columns: number of columns (12)
        x_size,y_size: figure size
        colorbar: show colorbar (False)
        y_pred: predicted classes (None)
        cm: Matplotlib olor map
    returns: 
        nothing
    """
    rows    = math.ceil(len(indices)/columns)
    fig=plt.figure(figsize=(columns*x_size, rows*(y_size+0.35)))
    n=1
    errors=0 
    if np.any(y_pred)==None:
        y_pred=y
    for i in indices:
        axs=fig.add_subplot(rows, columns, n)
        n+=1
        # ---- Shape is (lx,ly)
        if len(x[i].shape)==2:
            xx=x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape)==3:
            (lx,ly,lz)=x[i].shape
            if lz==1: 
                xx=x[i].reshape(lx,ly)
            else:
                xx=x[i]
        img=axs.imshow(xx,   cmap = cm, interpolation='lanczos')
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.set_yticks([])
        axs.set_xticks([])
        if y[i]!=y_pred[i]:
            axs.set_xlabel('{} ({})'.format(y_pred[i],y[i]))
            axs.xaxis.label.set_color('red')
            errors+=1
        else:
            axs.set_xlabel(y[i])
        if colorbar:
            fig.colorbar(img,orientation="vertical", shrink=0.65)
    plt.show()

def plot_image(x,cm='binary', figsize=(4,4)):
    (lx,ly,lz)=x.shape
    plt.figure(figsize=figsize)
    if lz==1:
        plt.imshow(x.reshape(lx,ly),   cmap = cm, interpolation='lanczos')
    else:
        plt.imshow(x.reshape(lx,ly,lz),cmap = cm, interpolation='lanczos')
    plt.show()


# -------------------------------------------------------------
# show_history
# -------------------------------------------------------------
#
def plot_history(history, figsize=(8,6)):
    """
    Show history
    args:
        history: history
        save_as: filename to save or None
    """
    # Accuracy 
    plt.figure(figsize=figsize)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Loss values
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()    


# -------------------------------------------------------------
# plot_confusion_matrix
# -------------------------------------------------------------
#
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          figsize=(12,8),
                          cmap="gist_heat_r",
                          vmin=0,
                          vmax=1,
                          xticks=5,yticks=5):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Args:
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
        title:        the text to display at the top of the matrix
        figsize:      Figure size (12,8)
        cmap:         color map (gist_heat_r)
        vmi,vmax:     Min/max 0 and 1
        
    """
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=figsize)
    sn.heatmap(cm, linewidths=1, linecolor="#ffffff",square=True, 
               cmap=cmap, xticklabels=xticks, yticklabels=yticks,
               vmin=vmin,vmax=vmax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
