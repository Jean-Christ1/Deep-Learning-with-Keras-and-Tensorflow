#!/bin/bash
#OAR -n Full convolutions
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=01:00:00
#OAR --stdout full_convolutions_%jobid%.out
#OAR --stderr full_convolutions_%jobid%.err
#OAR --project fidle

#---- With cpu
# use :
# OAR -l /nodes=1/core=32,walltime=02:00:00
# and add a 2>/dev/null to ipython xxx


# -----------------------------------------------
#         _           _       _
#        | |__   __ _| |_ ___| |__
#        | '_ \ / _` | __/ __| '_ \
#        | |_) | (_| | || (__| | | |
#        |_.__/ \__,_|\__\___|_| |_|
#                             Fidle at GRICAD
# -----------------------------------------------
#
# <!-- TITLE --> [BASH1] - OAR batch script
# <!-- DESC --> Bash script for OAR batch submission of GTSRB notebook 
# <!-- AUTHOR : Jean-Luc Parouty (CNRS/SIMaP) -->

CONDA_ENV=fidle
RUN_DIR=~/fidle/GTSRB
RUN_SCRIPT=./run/full_convolutions.py

# ---- Cuda Conda initialization
#
echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'
#
source /applis/environments/cuda_env.sh dahu 10.0
source /applis/environments/conda.sh
#
conda activate "$CONDA_ENV"

# ---- Run it...
#
cd $RUN_DIR
ipython $RUN_SCRIPT
