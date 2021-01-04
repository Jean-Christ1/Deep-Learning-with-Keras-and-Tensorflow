

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
VERSION = '0.6.1 DEV'

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
DEFAULT_SAVE_FIGS    = 'yes'

# ---- Catalog file, a json description of all notebooks
#
CATALOG_FILE   = '../fidle/log/catalog.json'

# ---- CI Done files, to keep track of finished notebooks -----------
# Used for continous integration
#
FINISHED_FILE  = '../fidle/log/finished.json'

# ---- CI Report
#
CI_REPORT      = '../fidle/log/ci_report.html'

# ---- Defaul mode (free|full|smart)
#      Overrided by env : FIDLE_RUNNING_MODE
#
DEFAULT_RUNNING_MODE = 'none'

# ---- CI Override parameters
#
GTSRB1_smart_scale      = 0.1
GTSRB1_smart_output_dir = './data'
GTSRB1_full_scale       = 1
GTSRB1_full_output_dir  = '{datasets_dir}/GTSRB/enhanced'

VAE6_smart_scale        = 0.05
VAE6_smart_image_size   = (128,128)
VAE6_smart_output_dir   = './data'
VAE6_full_scale         = 1
VAE6_full_image_size    = (192,160)
VAE6_full_output_dir    = '{datasets_dir}/celeba/enhanced'

VAE7_smart_image_size   = (128,128)
VAE7_smart_enhanced_dir = './data'
VAE7_full_image_size    = (192,160)
VAE7_full_enhanced_dir  = '{datasets_dir}/celeba/enhanced'

VAE8_smart_image_size   = (128,128)
VAE8_smart_enhanced_dir = './data'
VAE8_full_image_size    = (192,160)
VAE8_full_enhanced_dir  = '{datasets_dir}/celeba/enhanced'
