#!/bin/bash
#OAR -n VAE with CelebA
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=01:00:00
#OAR --stdout _batch/VAE_CelebA_%jobid%.out
#OAR --stderr _batch/VAE_CelebA_%jobid%.err
#OAR --project fidle

#---- For cpu
# use :
# OAR -l /nodes=1/core=32,walltime=01:00:00
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
# <!-- DESC --> Bash script for OAR batch submission of a notebook 
# <!-- AUTHOR : Jean-Luc Parouty (CNRS/SIMaP) -->

CONDA_ENV=fidle
RUN_DIR=~/fidle/VAE
RUN_IPYNB='06-VAE-with-CelebA-s.ipynb'
NODE_PREFIX=$(echo $HOSTNAME|sed 's/[0-9]*//g')

# ---- Cuda Conda initialization

echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'

source /applis/environments/cuda_env.sh $NODE_PREFIX 10.0
source /applis/environments/conda.sh

conda activate "$CONDA_ENV"

# ---- Run it...
#
cd $RUN_DIR

jupyter nbconvert --to notebook --execute "$RUN_IPYNB"
