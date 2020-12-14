
# -----------------------------------------------------------------------------
#       ____      _        _               ____        _ _     _
#      / ___|__ _| |_ __ _| | ___   __ _  | __ ) _   _(_) | __| | ___ _ __
#     | |   / _` | __/ _` | |/ _ \ / _` | |  _ \| | | | | |/ _` |/ _ \ '__|
#     | |__| (_| | || (_| | | (_) | (_| | | |_) | |_| | | | (_| |  __/ |
#      \____\__,_|\__\__,_|_|\___/ \__, | |____/ \__,_|_|_|\__,_|\___|_|
#                                  |___/                 Module catalog_builder
# -----------------------------------------------------------------------------
#
# A simple module to build the notebook catalog and update the README.

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import re
import sys, os, glob
import json
from collections import OrderedDict


def get_notebooks(directories, top_dir='..'):
    '''
    Return a list of notebooks from a given list of directories
    args:
        directories : list of directories
        top_dir : location of theses directories
    return:
        notebooks : notebooks filename list (without top_dir prefix)
    '''
    notebooks = []
    
    for d in directories:
        filenames = glob.glob( f'{top_dir}/{d}/*.ipynb')
        filenames.sort()
        notebooks.extend(filenames)

    notebooks = [ x.replace(f'{top_dir}/','') for x in notebooks]
    return notebooks


def get_infos(filename, top_dir='..'):
    '''
    Extract informations from a fidle notebook.
    Informations are dirname, basename, id, title, description and are extracted from comments tags in markdown.
    args:
        filename : Notebook filename
    return:
        dict : with infos.
    '''

    about={}
    about['dirname']     = os.path.dirname(filename)
    about['basename']    = os.path.basename(filename)
    about['id']          = '??'
    about['title']       = '??'
    about['description'] = '??'
    
    # ---- Read notebook
    #
    notebook = nbformat.read(f'{top_dir}/{filename}', nbformat.NO_CONVERT)
    
    # ---- Get id, title and desc tags
    #
    for cell in notebook.cells:

        if cell['cell_type'] == 'markdown':

            find = re.findall(r'<\!-- TITLE -->\s*\[(.*)\]\s*-\s*(.*)\n',cell.source)
            if find:
                about['id']    = find[0][0]
                about['title'] = find[0][1]

            find = re.findall(r'<\!-- DESC -->\s*(.*)\n',cell.source)
            if find:
                about['description']  = find[0]

    return about
                

def get_catalog(notebooks_list, top_dir='..'):
    '''
    Return an OrderedDict of notebooks attributes.
    Keys are notebooks id.
    args:
        notebooks_list : list of notebooks filenames
        top_dir : Location of theses notebooks
    return:
        OrderedDict : {<notebook id> : { description} }
    '''
    
    catalog = OrderedDict()

    for nb in notebooks_list:
        about = get_infos(nb, top_dir='..')
        id=about['id']
        catalog[id] = about

    return catalog
        

def tag(tag, text, document):
    debut  = f'<!-- {tag}_BEGIN -->'
    fin    = f'<!-- {tag}_END -->'

    output = re.sub(f'{debut}.*{fin}',f'{debut}\n{text}\n{fin}',document, flags=re.DOTALL)
    return output