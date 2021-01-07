

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
VERSION = '1.2b1 DEV'

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

# ---- Catalog file, a json description of all notebooks ------------
#
CATALOG_FILE   = '../fidle/log/catalog.json'

# ---- CI Done files, to keep track of finished notebooks -----------
# Used for continous integration
#
FINISHED_FILE  = '../fidle/log/finished.json'

# ---- CI Report ----------------------------------------------------
#
CI_REPORT      = '../fidle/log/ci_report.html'

# ---- CI Override parameters examples -----------------------------
#
# ---- Preparation of GTSRB dataset
# FIDLE_OVERRIDE_GTSRB1_scale      = 0.1
# FIDLE_OVERRIDE_GTSRB1_output_dir = './data'
# FIDLE_OVERRIDE_GTSRB1_scale       = 1
# FIDLE_OVERRIDE_GTSRB1_output_dir  = '{datasets_dir}/GTSRB/enhanced'

# # ---- Preparation of CelebA dataset
# FIDLE_OVERRIDE_VAE6_scale        = 0.2
# FIDLE_OVERRIDE_VAE6_image_size   = (128,128)
# FIDLE_OVERRIDE_VAE6_output_dir   = './data'
# FIDLE_OVERRIDE_VAE6_scale         = 1
# FIDLE_OVERRIDE_VAE6_image_size    = (192,160)
# FIDLE_OVERRIDE_VAE6_output_dir    = '{datasets_dir}/celeba/enhanced'

# # ---- Check CelebA dataset
# FIDLE_OVERRIDE_VAE7_image_size   = (128,128)
# FIDLE_OVERRIDE_VAE7_enhanced_dir = './data'
# FIDLE_OVERRIDE_VAE7_image_size    = (192,160)
# FIDLE_OVERRIDE_VAE7_enhanced_dir  = '{datasets_dir}/celeba/enhanced'

# # ---- VAE with CelebA
# FIDLE_OVERRIDE_VAE8_scale        = 1.
# FIDLE_OVERRIDE_VAE8_image_size   = (128,128)
# FIDLE_OVERRIDE_VAE8_enhanced_dir = './data'
# FIDLE_OVERRIDE_VAE8_scale         = 1.
# FIDLE_OVERRIDE_VAE8_image_size    = (192,160)
# FIDLE_OVERRIDE_VAE8_enhanced_dir  = '{datasets_dir}/celeba/enhanced'
