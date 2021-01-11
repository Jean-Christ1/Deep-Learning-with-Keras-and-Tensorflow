
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
from nbconvert.preprocessors import ExecutePreprocessor
import re
import yaml
from collections import OrderedDict
from IPython.display import display
import pandas as pd

sys.path.append('..')
import fidle.config as config
import fidle.cookindex as cookindex

start_time = None
end_time   = None

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

    config   = {'version':'1.0', 'output_tag':output_tag, 'save_figs':save_figs}
    profile  = { 'config':config }
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
        print(f'Catalog saved as {filename}')
        print('Entries : ',len(profile)-1)

        
def load_profile(filename):
    '''Load yaml profile'''
    with open(filename,'r') as fp:
        profile=yaml.load(fp, Loader=yaml.FullLoader)
        print(f'{filename} loaded.')
        print('Entries : ',len(profile)-1)
        return profile
    
    
def run_profile(profile, top_dir='..'):
    '''
    Récupère la liste des notebooks et des paramètres associés,
    décrit dans le profile, et pour chaque notebook :
    Positionner les variables d'environnement pour l'override
    Charger le notebook
    Exécuter celui-ci
    Sauvegarder le notebook résultat, avec son nom taggé.
    Params:
        profile : dict, profile d'éxécution
        top_dir : chemin relatif vers la racine fidle (..)
    '''

    # ---- My place
    #
    home = os.getcwd()
    
    # ---- Read profile
    #
    config    = profile['config']
    notebooks = profile
    del notebooks['config']
    
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
        chrono_start()
        update_ci_report(id, notebook_dir, notebook_src, notebook_out, start=True)

        # ---- Try to run...
        #
        print('    Run notebook...',end='')
        try:
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            ep.preprocess(notebook)
        except Exception as e:
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
        chrono_stop()        
        update_ci_report(id, notebook_dir, notebook_src, notebook_out, end=True, happy_end=happy_end)
        print('    Duration : ',chrono_delay() )
    
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
    
    
def chrono_start():
    global start_time
    start_time = datetime.datetime.now()

def chrono_stop():
    global end_time
    end_time = datetime.datetime.now()

def chrono_delay(in_seconds=False):
    global start_time, end_time
    delta = end_time - start_time
    if in_seconds:
        return round(delta.total_seconds(),2)
    else:
        delta = delta - datetime.timedelta(microseconds=delta.microseconds)
        return str(delta)



def reset_ci_report(verbose=True):
    data={}
    with open(config.CI_REPORT_JSON,'wt') as fp:
        json.dump(data,fp,indent=4)
    if verbose : print(f'Finished file has been reset.\n')
    
    
def update_ci_report(notebook_id, notebook_dir, notebook_src, notebook_out, start=False, end=False, happy_end=True):
    global start_time, end_time

    if not os.access(config.CI_REPORT_JSON, os.W_OK):
        reset_ci_report(verbose=True)
    
    # ---- Load it
    with open(config.CI_REPORT_JSON) as fp:
        report = json.load(fp)
        
    # ---- Update as a start
    if start is True:
        report[notebook_id]             = {}
        report[notebook_id]['dir']      = notebook_dir
        report[notebook_id]['src']      = notebook_src
        report[notebook_id]['out']      = notebook_out
        report[notebook_id]['start']    = start_time.strftime("%d/%m/%y %H:%M:%S")
        report[notebook_id]['end']      = ''
        report[notebook_id]['duration'] = 'Unfinished...'
        report[notebook_id]['state']    = 'Unfinished...'

    # ---- Update as an end
    if end is True:
        report[notebook_id]['end']      = end_time.strftime("%d/%m/%y %H:%M:%S")
        report[notebook_id]['duration'] = chrono_delay()
        report[notebook_id]['state']    = 'ok' if happy_end else 'ERROR'
        report[notebook_id]['out']      = notebook_out     # changeg in case of error

    # ---- Save it
    with open(config.CI_REPORT_JSON,'wt') as fp:
        json.dump(report,fp,indent=4)



def build_ci_report(display_output=True, save_html=True):
    
    # ---- Load ci report
    #
    with open(config.CI_REPORT_JSON) as infile:
        ci_report = json.load( infile )

    # ---- Create a nice DataFrame
    #
    df=pd.DataFrame(ci_report)
    df=df.transpose()
    df = df.rename_axis('id').reset_index()
    
    # ---- Change text columns, for nice html links
    #
    df['id']  = df.apply(lambda r: f"<a href='../{r['dir']}/{r['src']}'>{r['id']}</a>", axis=1)
    df['src'] = df.apply(lambda r: f"<a href='../{r['dir']}/{r['src']}'>{r['src']}</a>", axis=1)
    df['out'] = df.apply(lambda r: f"<a href='../{r['dir']}/{r['out']}'>{r['out']}</a>", axis=1)
        
    # ---- Add styles to be nice
    #
    styles = [
        dict(selector="td", props=[("font-size", "110%"), ("text-align", "left")]),
        dict(selector="th", props=[("font-size", "110%"), ("text-align", "left")])
    ]
    def still_pending(v):
        return 'background-color: OrangeRed; color:white' if v == 'ERROR' else ''

    output = df[df.columns.values].style.set_table_styles(styles).hide_index().applymap(still_pending)

    # ---- html report 
    #
    if save_html:
        html = _get_html_report(output)
        with open(config.CI_REPORT_HTML, "wt") as fp:
            fp.write(html)

    # ---- display output
    #
    if display_output:
        display(output)
    



def _get_html_report(output):
    with open('./img/00-Fidle-logo-01-80px.svg','r') as fp:
        logo = fp.read()

    now    = datetime.datetime.now().strftime("%A %-d %B %Y, %H:%M:%S")
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