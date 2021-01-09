
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
from IPython.display import display

sys.path.append('..')
import fidle.config as config

# -----------------------------------------------------------------------------
# To built README.md / README.ipynb
# -----------------------------------------------------------------------------
#    get_files      :  Get files lists
#    get_infos      :  Get infos about a entry
#    get_catalog    :  Get a catalog of all entries
# -----------------------------------------------------------------------------

def get_files(directories, top_dir='..'):
    '''
    Return a list of files from a given list of directories
    args:
        directories : list of directories
        top_dir : location of theses directories
    return:
        files : filenames list (without top_dir prefix)
    '''
    files = []
    regex = re.compile('.*==.+?==.*')

    for d in directories:
        notebooks = glob.glob( f'{top_dir}/{d}/*.ipynb')
        notebooks.sort()
        scripts   = glob.glob( f'{top_dir}/{d}/*.sh')
        scripts.sort()
        files.extend(notebooks)
        files.extend(scripts)
        
    files = [x for x in files if not regex.match(x)]
    files = [ x.replace(f'{top_dir}/','') for x in files]
    return files


def get_notebook_infos(filename, top_dir='..'):
    '''
    Extract informations from a fidle notebook.
    Informations are dirname, basename, id, title, description and are extracted from comments tags in markdown.
    args:
        filename : notebook filename
    return:
        dict : with infos.
    '''

    about={}
    about['id']          = '??'
    about['dirname']     = os.path.dirname(filename)
    about['basename']    = os.path.basename(filename)
    about['title']       = '??'
    about['description'] = '??'
    about['overrides']   = None
    
    # ---- Read notebook
    #
    notebook = nbformat.read(f'{top_dir}/{filename}', nbformat.NO_CONVERT)
    
    # ---- Get id, title and desc tags
    #
    overrides=[]
    for cell in notebook.cells:
     
        # ---- Find Index informations
        #
        if cell['cell_type'] == 'markdown':

            find = re.findall(r'<\!-- TITLE -->\s*\[(.*)\]\s*-\s*(.*)\n',cell.source)
            if find:
                about['id']    = find[0][0]
                about['title'] = find[0][1]

            find = re.findall(r'<\!-- DESC -->\s*(.*)\n',cell.source)
            if find:
                about['description']  = find[0]

        # ---- Find override informations
        #
        if cell['cell_type'] == 'code':
            
            # Try to find : override(...) call
            for m in re.finditer('override\((.+?)\)', cell.source):
                overrides.extend ( re.findall(r'\w+', m.group(1)) )

            # Try to find : run_dir=
            if re.search(r"\s*run_dir\s*?=", cell.source):
                overrides.append('run_dir')
                
    about['overrides']=overrides
    return about

    
    
def get_txtfile_infos(filename, top_dir='..'):
    '''
    Extract fidle  informations from a text file (script...).
    Informations are dirname, basename, id, title, description and are extracted from comments tags in document
    args:
        filename : file to analyze
    return:
        dict : with infos.
    '''

    about={}
    about['id']          = '??'
    about['dirname']     = os.path.dirname(filename)
    about['basename']    = os.path.basename(filename)
    about['title']       = '??'
    about['description'] = '??'
    about['overrides']   = []
    
    # ---- Read file
    #
    with open(f'{top_dir}/{filename}') as fp:
        text = fp.read()

    find = re.findall(r'<\!-- TITLE -->\s*\[(.*)\]\s*-\s*(.*)\n',text)
    if find:
        about['id']    = find[0][0]
        about['title'] = find[0][1]

    find = re.findall(r'<\!-- DESC -->\s*(.*)\n',text)
    if find:
        about['description']  = find[0]

    return about

              
def get_catalog(files_list=None, top_dir='..'):
    '''
    Return an OrderedDict of files attributes.
    Keys are file id.
    args:
        files_list : list of files filenames
        top_dir : Location of theses files
    return:
        OrderedDict : {<file id> : { description} }
    '''
       
    catalog = OrderedDict()

    # ---- Build catalog
    for file in files_list:
        about=None
        if file.endswith('.ipynb'): about = get_notebook_infos(file, top_dir='..')
        if file.endswith('.sh'):    about = get_txtfile_infos(file, top_dir='..')
        if about is None:
            print(f'** Warning : File [{file}] have no tags infomations...')
            continue
        id=about['id']
        catalog[id] = about
        
    return catalog
        

def tag(tag, text, document):
    debut  = f'<!-- {tag}_BEGIN -->'
    fin    = f'<!-- {tag}_END -->'

    output = re.sub(f'{debut}.*{fin}',f'{debut}\n{text}\n{fin}',document, flags=re.DOTALL)
    return output


def read_catalog():
    with open(config.CATALOG_FILE) as fp:
        catalog = json.load(fp)
    return catalog

# -----------------------------------------------------------------------------
# To built : CI Report
# -----------------------------------------------------------------------------
#    get_ci_report  :  Get report of finished files
# -----------------------------------------------------------------------------


def build_finished_report(display_output=True, save_html=True):
    
    # ---- Load catalog (notebooks descriptions)
    #
    with open(config.CATALOG_FILE) as fp:
        catalog = json.load(fp)

    # ---- Load finished file
    #
    with open(config.FINISHED_FILE) as infile:
        finished_files = json.load( infile )

    # ---- For each entry
    #
    col_id, col_repo, col_name, col_start, col_end, col_dur = [],[],[],[],[],[]
    for id,about in catalog.items():
        if id in finished_files.keys():
            finished  = finished_files[id]
            dirname   = about['dirname'] 
            basename  = about['basename'] 
            
            col_id.append(   f'<a href="../{dirname}/{basename}">{id}</a>'               )
            col_repo.append( dirname )
            col_name.append( f'<a href="../{dirname}/{basename}"><b>{basename}</b></a>'  )
            col_start.append( finished['start'] )
            col_end.append(   finished['end']   )
            col_dur.append(   finished['duration']   )
                   
    df = pd.DataFrame( {'id':col_id, 'Repo':col_repo, 'Name':col_name, 'Start':col_start, 'End':col_end, 'Duration':col_dur} )

    # ---- Add styles to be nice
    #
    styles = [
        dict(selector="td", props=[("font-size", "110%"), ("text-align", "left")]),
        dict(selector="th", props=[("font-size", "110%"), ("text-align", "left")])
    ]
    def still_pending(v):
        return 'background-color: OrangeRed; color:white' if v == 'Unfinished...' else ''

    output = df[df.columns.values].style.set_table_styles(styles).hide_index().applymap(still_pending)

    # ---- html report 
    #
    if save_html:
        html = _get_html_report(output)
        with open(config.FINISHED_REPORT, "wt") as fp:
            fp.write(html)

    # ---- display output
    #
    if display_output:
        display(output)
    



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