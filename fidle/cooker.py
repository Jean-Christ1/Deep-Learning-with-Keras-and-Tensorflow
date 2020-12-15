
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
import pandas as pd
from IPython.display import display, Markdown, HTML

import re
import sys, os, glob
import json
from datetime import datetime
from collections import OrderedDict

sys.path.append('..')
import fidle.config as config

# -----------------------------------------------------------------------------
# To built README.md / README.ipynb
# -----------------------------------------------------------------------------
#    get_notebooks  :  Get le notebooks lists
#    get_infos      :  Get infos about a notebooks
#    get_catalog    :  Get a catalog of all notebooks
# -----------------------------------------------------------------------------

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
    about['id']          = '??'
    about['dirname']     = os.path.dirname(filename)
    about['basename']    = os.path.basename(filename)
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


# -----------------------------------------------------------------------------
# To built : CI Report
# -----------------------------------------------------------------------------
#    get_ci_report  :  Get report of finished files
# -----------------------------------------------------------------------------


def get_ci_report():

    # ---- Load catalog (notebooks descriptions)
    #
    with open(config.CATALOG_FILE) as fp:
        catalog = json.load(fp)

    # ---- Load finished file
    #
    with open(config.FINISHED_FILE) as infile:
        dict_finished = json.load( infile )

    if dict_finished == {}:
        df=pd.DataFrame({}, columns=['id','name','start','end','duration'])
    else:
        df=pd.DataFrame(dict_finished).transpose()
        df.reset_index(inplace=True)
        df.rename(columns = {'index':'id'}, inplace=True)

    # ---- Add usefull html columns 
    #
    df['name']=''

    for index, row in df.iterrows():
        id = row['id']
        basename    = catalog[id]['basename']
        dirname     = catalog[id]['dirname']
        title       = catalog[id]['title']
        description = catalog[id]['description']
        row['id']   = f'<a href="../{dirname}/{basename}">{id}</a>'
        row['name'] = f'<a href="../{dirname}/{basename}"><b>{basename}</b></a>'

    columns=['id','name','start','end','duration']
    df=df[columns]

    # ---- Add styles to be nice
    #
    styles = [
        dict(selector="td", props=[("font-size", "110%"), ("text-align", "left")]),
        dict(selector="th", props=[("font-size", "110%"), ("text-align", "left")])
    ]
    def still_pending(v):
        return 'background-color: OrangeRed; color:white' if v == 'Unfinished...' else ''

    columns=['id','name','start','end','duration']

    output = df[columns].style.set_table_styles(styles).hide_index().applymap(still_pending)

    # ---- Get mail report 
    #
    html = _get_html_report(output)
    
    return df, output, html



def _get_html_report(output):
    with open('./img/00-Fidle-logo-01-80px.svg','r') as fp:
        logo = fp.read()

    now    = datetime.now().strftime("%A %-d %B %Y, %H:%M:%S")
    html = f"""\
    <html>
        <head><title>FIDLE - CI Report</title></head>
        <style>
            body{{
                  font-family: sans-serif;
            }}
            a{{
                color: SteelBlue;
                text-decoration:none;
            }}
            table{{      
                  border-collapse : collapse;
                  font-size : 80%
            }}
            td{{
                  border-style: solid;
                  border-width:  thin;
                  border-color:  lightgrey;
                  padding: 5px;
            }}
            .header{{ padding:20px 0px 0px 30px; }}
            .result{{ padding:10px 0px 20px 30px; }}
        </style>
        <body>
            <br>Hi,
            <p>Below is the result of the continuous integration tests of the Fidle project:</p>
            <div class="header"><b>Report date :</b> {now}</div>
            <div class="result">   
                {output.render()}
            </div>

            {logo}

            </body>
    </html>
    """
    return html