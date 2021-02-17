

# ==================================================================
#  ____                 _   _           _  __        __         _
# |  _ \ _ __ __ _  ___| |_(_) ___ __ _| | \ \      / /__  _ __| | __
# | |_) | '__/ _` |/ __| __| |/ __/ _` | |  \ \ /\ / / _ \| '__| |/ /
# |  __/| | | (_| | (__| |_| | (_| (_| | |   \ V  V / (_) | |  |   <
# |_|   |_|  \__,_|\___|\__|_|\___\__,_|_|    \_/\_/ \___/|_|  |_|\_\
#                                                      Configuration
# ==================================================================
# Few configuration stuffs for the Fidle practical work notebooks
# Jean-Luc Parouty 2020


# ---- Version -----------------------------------------------------
#
VERSION = '2.0.16'

# ---- Default notebook name ---------------------------------------
#
DEFAULT_NOTEBOOK_NAME = "Unknown"

# ---- Styles ------------------------------------------------------
#
FIDLE_MPLSTYLE = '../fidle/mplstyles/custom.mplstyle'
FIDLE_CSSFILE  = '../fidle/css/custom.css'

# ---- Save figs or not (yes|no)
#      Overided by env : FIDLE_SAVE_FIGS
#      
SAVE_FIGS    = False

# ---- Catalog file, a json description of all notebooks ------------
#
CATALOG_FILE    = '../fidle/logs/catalog.json'

# ---- CI report files ----------------------------------------------
#
CI_REPORT_JSON = '../fidle/logs/ci_report.json'
CI_REPORT_HTML = '../fidle/logs/ci_report.html'
CI_ERROR_FILE  = '../fidle/logs/ci_ERROR.txt'

# ------------------------------------------------------------------
