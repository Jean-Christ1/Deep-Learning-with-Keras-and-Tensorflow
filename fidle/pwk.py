
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
import shutil
from datetime import datetime
import itertools
import datetime, time
import json

import math
import numpy as np
from collections.abc import Iterable

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from IPython.display import display,Image,Markdown,HTML

import fidle.config as config


datasets_dir  = None
notebook_id   = None

_save_figs = False
_figs_dir  = './figs'
_figs_name = 'fig_'
_figs_id   = 0

_start_time = None
_end_time   = None

# -------------------------------------------------------------
# init_all
# -------------------------------------------------------------
#
def init(name=None, mplstyle=None, cssfile=None):
    global notebook_id
    global datasets_dir
    global _start_time
    
    # ---- Parameters
    #
    notebook_id = config.DEFAULT_NOTEBOOK_NAME if name is None else name

    if mplstyle is None:
        mplstyle = config.FIDLE_MPLSTYLE

    if cssfile  is None:
        cssfile  = config.FIDLE_CSSFILE
    
    # ---- Load matplotlib style and css
    #
    matplotlib.style.use(mplstyle)
    load_cssfile(cssfile)
    
    # ---- Create subdirs
    #
    mkdir('./run')
    
    # ---- Try to find where we are
    #
    datasets_dir = where_are_my_datasets()
    
    # ---- Update Keras cache
    #
    updated = update_keras_cache()
    
    # ---- Today and now
    #
    _start_time = datetime.datetime.now()
    
    # ---- Hello world
    print('\nFIDLE 2020 - Practical Work Module')
    print('Version              :', config.VERSION)
    print('Notebook id          :', notebook_id)
    print('Run time             :', _start_time.strftime("%A %-d %B %Y, %H:%M:%S"))
    print('TensorFlow version   :', tf.__version__)
    print('Keras version        :', tf.keras.__version__)
    print('Datasets dir         :', datasets_dir)
    print('Update keras cache   :',updated)
    
    update_finished_file(start=True)

    return datasets_dir

# ------------------------------------------------------------------
# Update keras cache
# ------------------------------------------------------------------
# Try to sync ~/.keras/cache with datasets/keras_cache
# because sometime, we cannot access to internet... (IDRIS..)
#
def update_keras_cache():
    updated = False
    if os.path.isdir(f'{datasets_dir}/keras_cache'):
        from_dir = f'{datasets_dir}/keras_cache/*.*'
        to_dir   = os.path.expanduser('~/.keras/datasets')
        mkdir(to_dir)
        for pathname in glob.glob(from_dir):
            filename=os.path.basename(pathname)
            destname=f'{to_dir}/{filename}'
            if not os.path.isfile(destname):
                shutil.copy(pathname, destname)
                updated=True
    return updated

# ------------------------------------------------------------------
# Where are my datasets ?
# ------------------------------------------------------------------
#
def where_are_my_datasets():
        
    datasets_dir = os.getenv('FIDLE_DATASETS_DIR', False)

    if datasets_dir is False :
        display_md('## ATTENTION !!\n----')
        print('Le dossier datasets sont introuvable\n')
        print('Pour que les notebooks puissent les localiser, vous devez :\n')
        print('         1/ Récupérer le dossier datasets')
        print('            Une archive (datasets.tar) est disponible via le repository Fidle.\n')
        print("         2/ Préciser la localisation de ce dossier datasets via la variable")
        print("            d'environnement : FIDLE_DATASETS_DIR.\n")
        print('Exemple :')
        print("   Dans votre fichier .bashrc :")
        print('   export FIDLE_DATASETS_DIR=~/datasets')
        display_md('----')
        assert False, 'datasets folder not found, please set FIDLE_DATASETS_DIR env var.'
    else:
        return datasets_dir

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
                fontsize=20, save_as='auto'):
    """
    Show some images in a grid, with legends
    args:
        x             : images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y             : real classes or labels or None (None)
        indices       : indices of images to show or None for all (None)
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1)
        colorbar      : show colorbar (False)
        y_pred        : predicted classes (None)
        cm            : Matplotlib color map (binary)
        y_padding     : Padding / rows (0.35)
        spines_alpha  : Spines alpha (1.)
        font_size     : Font size in px (20)
        save_as       : Filename to use if save figs is enable ('auto')
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
    save_fig(save_as)
    plt.show()

    
def plot_image(x,cm='binary', figsize=(4,4),save_as='auto'):
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
    save_fig(save_as)
    plt.show()


# -------------------------------------------------------------
# show_history
# -------------------------------------------------------------
#
def plot_history(history, figsize=(8,6), 
                 plot={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']},
                 save_as='auto'):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    fig_id=0
    for title,curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        if save_as=='auto':
            figname='auto'
        else:
            figname=f'{save_as}_{fig_id}'
            fig_id+=1
        save_fig(figname)
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
                          xticks=5,yticks=5,
                          annot=True,
                          save_as='auto'):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Note:bug in matplotlib 3.1.1

    Args:
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
        title:        the text to display at the top of the matrix
        figsize:      Figure size (12,8)
        cmap:         color map (gist_heat_r)
        vmi,vmax:     Min/max 0 and 1
        annot:        Annotation or just colors (True)
        
    """
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=figsize)
    sn.heatmap(cm, linewidths=1, linecolor="#ffffff",square=True, 
               cmap=cmap, xticklabels=xticks, yticklabels=yticks,
               vmin=vmin,vmax=vmax,annot=annot)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    save_fig(save_as)
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
    
    
def plot_donut(values, labels, colors=["lightsteelblue","coral"], figsize=(6,6), title=None, save_as='auto'):
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
    save_fig(save_as)
    plt.show()
    

    
def plot_multivariate_serie(sequence, labels=None, predictions=None, only_features=None,
                            columns=3, width=5,height=4,wspace=0.3,hspace=0.2,
                            save_as='auto', time_dt=1):
    
    sequence_len = len(sequence)
    features_len = sequence.shape[1]
    if only_features is None : only_features=range(features_len)
    if labels is None        : labels=range(features_len)
    
    t  = np.arange(sequence_len)    
    if predictions is None:
        dt = 0
    else:
        dt = len(predictions)
        sequence_with_pred = sequence.copy()
        sequence_with_pred[-dt:]=predictions

    rows = math.ceil(features_len/columns)
    fig  = plt.figure(figsize=(columns*width, rows*height))
    fig.subplots_adjust(wspace=0.3,hspace=0.2)
    n=1
    for i in only_features:
        ax=fig.add_subplot(rows, columns, n)
        ax.plot(t[:-dt],       sequence[:-dt,i],    '-',  linewidth=1,  color='steelblue', label=labels[i])
        ax.plot(t[:-dt],       sequence[:-dt,i],    'o',  markersize=4, color='steelblue')
        ax.plot(t[-dt-1:], sequence[-dt-1:,i],'--o', linewidth=1, fillstyle='none',  markersize=6, color='steelblue')
        if predictions is not None:
            ax.plot(t[-dt-1:],     sequence_with_pred[-dt-1:,i],     '--',  linewidth=1, fillstyle='full',  markersize=6, color='red')
            ax.plot(t[-dt:],       predictions[:,i],     'o',  linewidth=1, fillstyle='full',  markersize=6, color='red')

        ax.legend(loc="upper left")
        n+=1
    save_fig(save_as)
    plt.show()

 
    
    
    
def set_save_fig(save=True, figs_dir='./figs', figs_name='fig_', figs_id=0):
    """
    Set save_fig parameters
    Default figs name is <figs_name><figs_id>.{png|svg}
    args:
        save      : Boolean, True to save figs (True)
        figs_dir  : Path to save figs (./figs)
        figs_name : Default basename for figs (figs_)
        figs_id   : Start id for figs name (0)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    _save_figs = save
    _figs_dir  = figs_dir
    _figs_name = figs_name
    _figs_id   = figs_id
    print(f'Save figs            : {_save_figs}')
    print(f'Path figs            : {_figs_dir}')
    
    
def save_fig(filename='auto', png=True, svg=False):
    """
    Save current figure
    args:
        filename : Image filename ('auto')
        png      : Boolean. Save as png if True (True)
        svg      : Boolean. Save as svg if True (False)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    if not _save_figs : return
    mkdir(_figs_dir)
    if filename=='auto': 
        path=f'{_figs_dir}/{_figs_name}{_figs_id:02d}'
    else:
        path=f'{_figs_dir}/{filename}'
    if png : plt.savefig( f'{path}.png')
    if svg : plt.savefig( f'{path}.png')
    if filename=='auto': _figs_id+=1
    

def subtitle(t):
    display(Markdown(f'<br>**{t}**'))
    
def display_md(md_text):
    display(Markdown(md_text))
    
def display_img(img):
    display(Image(img))
    
def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))

# Return human delay like 01:14:28 543ms
def hdelay_ms(td):
    sec = td.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh*3600 - mm*60
    ms = (sec - int(sec))*1000
    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

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
    
     
        
def np_print(*args, format={'float': '{:6.3f}'.format}):
    with np.printoptions(formatter=format):
        for a in args:
            print(a)

            
def check_finished_file():
    if not os.access(config.FINISHED_FILE, os.W_OK):
        print("\n** Error : Cannot access finished file in write mode for reset...")
        print(f'** Finished file should be at : {config.FINISHED_FILE}\n')
        return False
    return True
    
    
def reset_finished_file():
    if check_finished_file()==False : return
    data={}
    # ---- Save it
    with open(config.FINISHED_FILE,'wt') as fp:
        json.dump(data,fp,indent=4)
    print(f'Finished file has been reset.\n')
    
    
def update_finished_file(start=False, end=False):
    
    # ---- No writable finished file ?
    if check_finished_file() is False : return
    
    # ---- Load it
    with open(config.FINISHED_FILE) as fp:
        data = json.load(fp)
        
    # ---- Update as a start
    if start is True:
        data[notebook_id]             = {}
        data[notebook_id]['path']     = os.getcwd()
        data[notebook_id]['start']    = _start_time.strftime("%A %-d %B %Y, %H:%M:%S")
        data[notebook_id]['end']      = ''
        data[notebook_id]['duration'] = 'Pending...'

    # ---- Update as an end
    if end is True:
        data[notebook_id]['end']      = _end_time.strftime("%A %-d %B %Y, %H:%M:%S")
        data[notebook_id]['duration'] = hdelay_ms(_end_time - _start_time)

    # ---- Save it
    with open(config.FINISHED_FILE,'wt') as fp:
        json.dump(data,fp,indent=4)
    
     
def end():
    global _end_time
    _end_time = datetime.datetime.now()
    
    update_finished_file(end=True)
    
    print('End time is :', time.strftime("%A %-d %B %Y, %H:%M:%S"))
    print('Duration is :', hdelay_ms(_end_time - _start_time))
    print('This notebook ends here')
     
     
