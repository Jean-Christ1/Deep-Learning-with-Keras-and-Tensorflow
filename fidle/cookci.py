
# -----------------------------------------------------------------------------
#                         ____                 _       ____ _
#                       / ___|___   ___   ___| | __  / ___(_)
#                      | |   / _ \ / _ \ / __| |/ / | |   | |
#                      | |__| (_) | (_) | (__|   <  | |___| |
#                      \____\___/ \___/ \___|_|\_\  \____|_|
#
#                                           Fidle mod for continous integration
# -----------------------------------------------------------------------------
#
# A simple module to run all notebooks with parameters overriding
# Jean-Luc Parouty 2021

import sys,os
import json
import datetime, time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from asyncio import CancelledError
import re
import yaml
from collections import OrderedDict
from IPython.display import display,Image,Markdown,HTML
import pandas as pd

sys.path.append('..')
import fidle.config as config
import fidle.cookindex as cookindex

VERSION = '1.0'

start_time = {}
end_time   = {}

def get_ci_profile(catalog=None, output_tag='==done==', save_figs=True):
    '''
    Return a profile for continous integration.
    Ce profile contient une liste des notebooks avec les paramètres modifiables.
    Il peut être modifié et sauvegardé, puis être utilisé pour lancer l'éxécution
    des notebooks.
    params:
        catalog : Notebooks catalog. if None (default), load config.CATALOG_FILE
        output_tag  : tag name of generated notebook
        save_figs : save figs or not for generated notebooks (True)
    return:
        profile : dict with run parameters
    '''
    
    if catalog is None:
        catalog = cookindex.read_catalog()

    metadata   = { 'version'     : '1.0', 
                   'output_tag'  : output_tag, 
                   'save_figs'   : save_figs, 
                   'description' : 'Default generated profile'}
    profile  = { '_metadata_':metadata }
    for id, about in catalog.items():
        
        id        = about['id']
        title     = about['title']
        dirname   = about['dirname']
        basename  = about['basename']
        overrides = about.get('overrides',None)
    
        notebook = {}
        notebook['notebook_dir'] = dirname
        notebook['notebook_src'] = basename
        notebook['notebook_out'] = 'default'
        if len(overrides)>0:
            notebook['overrides']={ name:'default' for name in overrides }
                    
        profile[id]=notebook
        
    return profile


def save_profile(profile, filename):
    '''Save profile in yaml format'''
    with open(filename,'wt') as fp:
        yaml.dump(profile, fp, sort_keys=False)
        print(f'Profile saved as {filename}')
        print('Entries : ',len(profile)-1)

        
def load_profile(filename):
    '''Load yaml profile'''
    with open(filename,'r') as fp:
        profile=yaml.load(fp, Loader=yaml.FullLoader)
        print(f'\nLoad profile :{filename}')
        print('    Entries : ',len(profile)-1)
        return profile
    
    
def run_profile(profile_name, report_name=None, top_dir='..'):
    '''
    Récupère la liste des notebooks et des paramètres associés,
    décrit dans le profile, et pour chaque notebook :
    Positionner les variables d'environnement pour l'override
    Charger le notebook
    Exécuter celui-ci
    Sauvegarder le notebook résultat, avec son nom taggé.
    Params:
        profile_name : nom du profile d'éxécution
        report_name : Nom du rapport json généré
        top_dir : chemin relatif vers la racine fidle (..)
    '''

    print('\nRun profile session - FIDLE 2021')
    print(f'Version : {VERSION}')
    
    chrono_start('main')
    
    # ---- Retrieve profile
    #
    profile   = load_profile(profile_name)
    config    = profile['_metadata_']
    notebooks = profile
    del notebooks['_metadata_']   
    
    # ---- Create new ci_report
    #
    metadata = config
    metadata['host']    = os.uname()[1]
    metadata['profile'] = profile_name
    if report_name is None:
        report_name = config.CI_REPORT_JSON
    report_name = os.path.abspath(report_name)
    create_ci_report(report_name, metadata)
    
    # ---- My place
    #
    home = os.getcwd()
        
    # ---- Save figs or not ?
    #
    os.environ['FIDLE_SAVE_FIGS']=str(config['save_figs'])

    # ---- For each notebook
    #
    for id,about in notebooks.items():
        
        print(f'\nNotebook : {id}')

        # ---- Get notebook infos ---------------------------------------------
        #
        notebook_dir = about['notebook_dir']
        notebook_src = about['notebook_src']
        notebook_out = about['notebook_out']
        overrides    = about.get('overrides',None)

        # ---- notebook_out (Default)
        #
        if notebook_out=='default':
            notebook_out = notebook_src.replace('.ipynb', config['output_tag']+'.ipynb')
                
        # ---- Override ------------------------------------------------------- 
        
        to_unset=[]
        if isinstance(overrides,dict):
            print('    set overrides :')
            for name,value in overrides.items():
                # ---- Default : no nothing
                if value=='default' : continue
                #  ---- Set env
                env_name  = f'FIDLE_OVERRIDE_{id.upper()}_{name}'
                env_value = str(value)
                os.environ[env_name] = env_value
                # ---- For cleaning
                to_unset.append(env_name)
                # ---- Fine :(-)
                print(f'       {env_name:38s} = {env_value}')
    
        # ---- Run it ! -------------------------------------------------------
    
        # ---- Go to the right place
        #
        os.chdir(f'{top_dir}/{notebook_dir}')
        notebook = nbformat.read( f'{notebook_src}', nbformat.NO_CONVERT)

        # ---- Top chrono
        #
        chrono_start('nb')
        update_ci_report(report_name, id, notebook_dir, notebook_src, notebook_out, start=True)
        
        # ---- Try to run...
        #
        print('    Run notebook...',end='')
        try:
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            ep.preprocess(notebook)
        except CellExecutionError as e:
            happy_end = False
            notebook_out = notebook_src.replace('.ipynb', '==ERROR==.ipynb')
            print('\n   ','*'*60)
            print( '    ** AAARG.. An error occured : ',type(e).__name__)
            print(f'    ** See notebook :  {notebook_out} for details.')
            print('   ','*'*60)
        else:
            happy_end = True
            print('..done.')

        # ---- Top chrono
        #
        chrono_stop('nb')        
        update_ci_report(report_name, id, notebook_dir, notebook_src, notebook_out, end=True, happy_end=happy_end)
        print('    Duration : ',chrono_get_delay('nb') )
    
        # ---- Save notebook
        #
        with open( f'{notebook_out}', mode="w", encoding='utf-8') as fp:
            nbformat.write(notebook, fp)
        print('    Saved as : ',notebook_out)
    
        # ---- Back to home and clean
        #
        os.chdir(home)
        for env_name in to_unset:
            del os.environ[env_name]

    # ---- End of running
    chrono_stop('main')
    print('\nEnd of running process')
    print('    Duration :', chrono_get_delay('main'))
    complete_ci_report(report_name)
    
    
def chrono_start(id='default'):
    global start_time
    start_time[id] = datetime.datetime.now()
        
def chrono_stop(id='default'):
    global end_time
    end_time[id] = datetime.datetime.now()

def chrono_get_delay(id='default', in_seconds=False):
    global start_time, end_time
    delta = end_time[id] - start_time[id]
    if in_seconds:
        return round(delta.total_seconds(),2)
    else:
        delta = delta - datetime.timedelta(microseconds=delta.microseconds)
        return str(delta)

def chrono_get_start(id='default'):
    global start_time
    return start_time[id].strftime("%d/%m/%y %H:%M:%S")

def chrono_get_end(id='default'):
    global end_time
    return end_time[id].strftime("%d/%m/%y %H:%M:%S")

def reset_chrono():
    global start_time, end_time
    start_time, end_time = {},{}
    

def create_ci_report(filename, metadata, verbose=True):
    metadata['start']=chrono_get_start('main')
    data={ '_metadata_':metadata }
    with open(filename,'wt') as fp:
        json.dump(data,fp,indent=4)
    if verbose : print(f'\nCreate new ci report : {filename}')
    
def complete_ci_report(filename, verbose=True):
    with open(filename) as fp:
        report = json.load(fp)
    report['_metadata_']['end']      = chrono_get_end('main')
    report['_metadata_']['duration'] = chrono_get_delay('main')
    with open(filename,'wt') as fp:
        json.dump(report,fp,indent=4)
    if verbose : print(f'\nComplete ci report : {filename}')
        
def update_ci_report(filename, notebook_id, notebook_dir, notebook_src, notebook_out, start=False, end=False, happy_end=True):
    global start_time, end_time
    
    # ---- Load it
    with open(filename) as fp:
        report = json.load(fp)
        
    # ---- Update as a start
    if start is True:
        report[notebook_id]              = {}
        report[notebook_id]['dir']       = notebook_dir
        report[notebook_id]['src']       = notebook_src
        report[notebook_id]['out']       = notebook_out
        report[notebook_id]['start']     = chrono_get_start('nb')
        report[notebook_id]['end']       = ''
        report[notebook_id]['duration']  = 'Unfinished...'
        report[notebook_id]['state']     = 'Unfinished...'
        report['_metadata_']['end']      = 'Unfinished...'
        report['_metadata_']['duration'] = 'Unfinished...'


    # ---- Update as an end
    if end is True:
        report[notebook_id]['end']       = chrono_get_end('nb')
        report[notebook_id]['duration']  = chrono_get_delay('nb')
        report[notebook_id]['state']     = 'ok' if happy_end else 'ERROR'
        report[notebook_id]['out']       = notebook_out     # changeg in case of error

    # ---- Save it
    with open(filename,'wt') as fp:
        json.dump(report,fp,indent=4)



def build_ci_report(report_name=None, display_output=True, save_html=True):
    
    # ---- Load ci report
    #
    if report_name is None:
        report_name = config.CI_REPORT_JSON
    with open(report_name) as infile:
        ci_report = json.load( infile )

    # ---- metadata
    #
    metadata=ci_report['_metadata_']
    del ci_report['_metadata_']
    
    metadata_md=''
    metadata_html=''
    for name,value in metadata.items():
        metadata_md   += f'**{name.title()}** : {value}  \n'
        metadata_html += f'<b>{name.title()}</b> : {value}  <br>\n'
    
    # ---- Create a nice DataFrame
    #
    df=pd.DataFrame(ci_report)
    df=df.transpose()
    df = df.rename_axis('id').reset_index()

    # ---- Few styles to be nice
    #
    styles = [
        dict(selector="td", props=[("font-size", "110%"), ("text-align", "left")]),
        dict(selector="th", props=[("font-size", "110%"), ("text-align", "left")])
    ]
    def still_pending(v):
        return 'background-color: OrangeRed; color:white' if v == 'ERROR' else ''

    # ---- Links version : display
    #
    if display_output:
        
        ddf=df.copy()
        ddf['id']  = ddf.apply(lambda r: f"<a href='../{r['dir']}/{r['src']}'>{r['id']}</a>", axis=1)
        ddf['src'] = ddf.apply(lambda r: f"<a href='../{r['dir']}/{r['src']}'>{r['src']}</a>", axis=1)
        ddf['out'] = ddf.apply(lambda r: f"<a href='../{r['dir']}/{r['out']}'>{r['out']}</a>", axis=1)
        ddf.columns = [x.title() for x in ddf.columns]

        output = ddf[ddf.columns.values].style.set_table_styles(styles).hide_index().applymap(still_pending)
        display(Markdown('### About :'))
        display(Markdown(metadata_md))
        display(Markdown('### Details :'))
        display(output)

    # ---- Basic version : html report 
    #
    if save_html:
        
        df.columns = [x.title() for x in df.columns]
        output = df[df.columns.values].style.set_table_styles(styles).hide_index().applymap(still_pending)

        html = _get_html_report(metadata_html, output)
        with open(config.CI_REPORT_HTML, "wt") as fp:
            fp.write(html)
        display(Markdown('<br>HTML report saved as : [./logs/ci_report.html](./logs/ci_report.html)'))
            



def _get_html_report(metadata_html, output):
    with open('./img/00-Fidle-logo-01-80px.svg','r') as fp:
        logo = fp.read()

    html = f"""\
    <html>
        <head><title>FIDLE - CI Report</title></head>
        <body>
        <style>
            body{{
                  font-family: sans-serif;
            }}
            div.title{{ 
                font-size: 1.2em;
                font-weight: bold;
                padding: 15px 0px 10px 0px; }}
            a{{
                color: SteelBlue;
                text-decoration:none;
            }}
            table{{      
                  border-collapse : collapse;
                  font-size : 0.9em;
            }}
            td{{
                  border-style: solid;
                  border-width:  thin;
                  border-color:  lightgrey;
                  padding: 5px;
            }}
            .metadata{{ padding: 10px 0px 10px 30px; font-size: 0.9em; }}
            .result{{ padding: 10px 0px 10px 30px; }}
        </style>
            <br>Hi,
            <p>Below is the result of the continuous integration tests of the Fidle project:</p>
            <div class='title'>About :</div>
            <div class="metadata">{metadata_html}</div>
            <div class='title'>Details :</div>
            <div class="result">{output.render()}</div>

            {logo}

            </body>
    </html>
    """
    return html