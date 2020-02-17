
# ==================================================================
#  ____                 _   _           _  __        __         _
# |  _ \ _ __ __ _  ___| |_(_) ___ __ _| | \ \      / /__  _ __| | __
# | |_) | '__/ _` |/ __| __| |/ __/ _` | |  \ \ /\ / / _ \| '__| |/ /
# |  __/| | | (_| | (__| |_| | (_| (_| | |   \ V  V / (_) | |  |   <
# |_|   |_|  \__,_|\___|\__|_|\___\__,_|_|    \_/\_/ \___/|_|  |_|\_\
#                                                        module pwk                                   
# ==================================================================
# A simple module to host some common functions for practical work
# Jean-Luc Parouty 2020

import os
import glob
from datetime import datetime
import itertools
import datetime, time

import math
import numpy as np
from collections.abc import Iterable

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sn     #IDRIS : module en cours d'installation

from IPython.display import display,Markdown,HTML

VERSION='0.2.9'


# -------------------------------------------------------------
# init_all
# -------------------------------------------------------------
#
def init(mplstyle='../fidle/mplstyles/custom.mplstyle', cssfile='../fidle/css/custom.css'):
    global VERSION
    # ---- matplotlib and css
    matplotlib.style.use(mplstyle)
    load_cssfile(cssfile)
    # ---- Hello world
#     now = datetime.datetime.now()
    print('\nFIDLE 2020 - Practical Work Module')
    print('Version              :', VERSION)
    print('Run time             : {}'.format(time.strftime("%A %-d %B %Y, %H:%M:%S")))
    print('TensorFlow version   :',tf.__version__)
    print('Keras version        :',tf.keras.__version__)
          
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
    """
    Shuffle a dataset (x,y)
    args:
        x,y : dataset
    return:
        x,y mixed
    """
    assert (len(x) == len(y)), "x and y must have same size"
    p = np.random.permutation(len(x))
    return x[p], y[p]


def update_progress(what,i,imax, redraw=False):
    """
    Display a text progress bar, as :
    My progress bar : ############# 34%
    args:
        what  : Progress bas name
        i     : Current progress
        imax  : Max value for i
    return:
        nothing
    """
    bar_length = min(40,imax)
    if (i%int(imax/bar_length))!=0 and i<imax and not redraw:
        return
    progress  = float(i/imax)
    block     = int(round(bar_length * progress))
    endofline = '\r' if progress<1 else '\n'
    text = "{:16s} [{}] {:>5.1f}% of {}".format( what, "#"*block+"-"*(bar_length-block), progress*100, imax)
    print(text, end=endofline)

    
def rmax(l):
    """
    Recursive max() for a given iterable of iterables
    Should be np.array of np.array or list of list, etc.
    args:
        l : Iterable of iterables
    return: 
        max value
    """
    maxi = float('-inf')
    for item in l:
        if isinstance(item, Iterable):
            t = rmax(item)
        else:
            t = item
        if t > maxi:
            maxi = t
    return maxi

def rmin(l):
    """
    Recursive min() for a given iterable of iterables
    Should be np.array of np.array or list of list, etc.
    args:
        l : Iterable of iterables
    return: 
        min value
    """
    mini = float('inf')
    for item in l:
        if isinstance(item, Iterable):
            t = rmin(item)
        else:
            t = item
        if t < mini:
            mini = t
    return mini

# -------------------------------------------------------------
# show_images
# -------------------------------------------------------------
#
def plot_images(x,y=None, indices='all', columns=12, x_size=1, y_size=1,
                colorbar=False, y_pred=None, cm='binary',y_padding=0.35, spines_alpha=1,
                fontsize=20):
    """
    Show some images in a grid, with legends
    args:
        x: images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y: real classes or labels or None (None)
        indices: indices of images to show or None for all (None)
        columns: number of columns (12)
        x_size,y_size: figure size (1), (1)
        colorbar: show colorbar (False)
        y_pred: predicted classes (None)
        cm: Matplotlib color map (binary)
    returns: 
        nothing
    """
    if indices=='all': indices=range(len(x))
    draw_labels = (y is not None)
    draw_pred   = (y_pred is not None)
    rows        = math.ceil(len(indices)/columns)
    fig=plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n=1
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
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i],fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i]!=y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})',fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i],fontsize=fontsize)
        if colorbar:
            fig.colorbar(img,orientation="vertical", shrink=0.65)
    plt.show()

    
def plot_image(x,cm='binary', figsize=(4,4)):
    """
    Draw a single image.
    Image shape can be (lx,ly), (lx,ly,1) or (lx,ly,n)
    args:
        x       : image as np array
        cm      : color map ('binary')
        figsize : fig size (4,4)
    """
    # ---- Shape is (lx,ly)
    if len(x.shape)==2:
        xx=x
    # ---- Shape is (lx,ly,n)
    if len(x.shape)==3:
        (lx,ly,lz)=x.shape
        if lz==1: 
            xx=x.reshape(lx,ly)
        else:
            xx=x
    # ---- Draw it
    plt.figure(figsize=figsize)
    plt.imshow(xx,   cmap = cm, interpolation='lanczos')
    plt.show()


# -------------------------------------------------------------
# show_history
# -------------------------------------------------------------
#
def plot_history(history, figsize=(8,6), 
                  plot={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']}):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    for title,curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        plt.show()

    
    
# -------------------------------------------------------------
# plot_confusion_matrix
# -------------------------------------------------------------
# Bug in Matplotlib 3.1.1
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
    Note:bug in matplotlib 3.1.1

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
               vmin=vmin,vmax=vmax,annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()


    
def display_confusion_matrix(y_true,y_pred,labels=None,color='green',
                             font_size='12pt', title="#### Confusion matrix is :"):
    """
    Show a confusion matrix for a predictions.
    see : sklearn.metrics.confusion_matrix

    Args:
        y_true        Real classes
        y_pred        Predicted classes
        labels        List of classes to show in the cm
        color:        Color for the palette (green)
        font_size:    Values font size 
        title:        the text to display at the top of the matrix        
    """
    assert (labels!=None),"Label must be set"
    
    if title != None :  display(Markdown(title)) 
    
    cm = confusion_matrix( y_true,y_pred, normalize="true", labels=labels)
    df=pd.DataFrame(cm)

    cmap = sn.light_palette(color, as_cmap=True)
    df.style.set_properties(**{'font-size': '20pt'})
    display(df.style.format('{:.2f}') \
            .background_gradient(cmap=cmap)
            .set_properties(**{'font-size': font_size}))
    
    
def plot_donut(values, labels, colors=["lightsteelblue","coral"], figsize=(6,6), title=None):
    """
    Draw a donut
    args:
        values   : list of values
        labels   : list of labels
        colors   : list of color (["lightsteelblue","coral"])
        figsize  : size of figure ( (6,6) )
    return:
        nothing
    """
    # ---- Title or not
    if title != None :  display(Markdown(title))
    # ---- Donut
    plt.figure(figsize=figsize)
    # ---- Draw a pie  chart..
    plt.pie(values, labels=labels, 
            colors = colors, autopct='%1.1f%%', startangle=70, pctdistance=0.85,
            textprops={'fontsize': 18},
            wedgeprops={"edgecolor":"w",'linewidth': 5, 'linestyle': 'solid', 'antialiased': True})
    # ---- ..with a white circle
    circle = plt.Circle((0,0),0.70,fc='white')
    ax = plt.gca()
    ax.add_artist(circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()
    
def display_md(md_text):
    display(Markdown(md_text))
    
def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))

def hsize(num, suffix='o'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f} Y{suffix}'

def load_cssfile(cssfile):
    if cssfile is None: return
    styles = open(cssfile, "r").read()
    display(HTML(styles))
    
    
def good_place( places={'SOMEWHERE':'/tmp'} ):
    for place_name, place_dir in places.items():
        if os.path.isdir(place_dir):
            print(f'Well, we should be at {place_name} !')
            print(f'We are going to use: {place_dir}')
            return place_name,place_dir

    print('** Attention : No expected folder exists in this environment..')
    assert False, 'No expected folder exists in this environment..'
     
     
     
     
     
     
     
     
