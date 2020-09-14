

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

import os

# ---- Current version ---------------------------------------------
#
VERSION = '0.58'

# ---- Locations ---------------------------------------------------
#
# A list of locations where this notebooks can be executed, with the
# location of the datasets folder.
# You can complete this list by adding specifics locations.
#
# Syntax is : 
#   locations = {  <location name>:<datasets path> , ...}
#
# Example : 
#   locations = { 'My laptop':'/usr/local/datasets' }
#
# This locations are defaults locations :
#
locations = { 'Fidle at GRICAD' : f"{os.getenv('SCRATCH_DIR',   'nowhere' )}/PROJECTS/pr-fidle/datasets",
              'Fidle at IDRIS'  : "/gpfswork/rech/mlh/uja62cb/datasets",
              'Fidle at HOME'   : f"{os.getenv('HOME',          'nowhere' )}/datasets"}

# ------------------------------------------------------------------
