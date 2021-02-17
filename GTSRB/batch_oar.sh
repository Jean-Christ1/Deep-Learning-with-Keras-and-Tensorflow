#!/bin/bash
#OAR -n Full convolutions
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=01:00:00
#OAR --stdout full_convolutions_%jobid%.out
#OAR --stderr full_convolutions_%jobid%.err
#OAR --project fidle

#---- Note for cpu, set :
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
# <!-- TITLE --> [GTSRB10] - OAR batch script submission
# <!-- DESC -->  Bash script for an OAR batch submission of an ipython code
# <!-- AUTHOR : Jean-Luc Parouty (CNRS/SIMaP) -->

# ==== Notebook parameters =========================================

CONDA_ENV='fidle'
NOTEBOOK_DIR="~/fidle/GTSRB"

SCRIPT_IPY="05-Full-convolutions.py"

# ---- Environment vars used to override notebook/script parameters
#
export FIDLE_OVERRIDE_GTSRB5_run_dir="./run/GTSRB5"
export FIDLE_OVERRIDE_GTSRB5_datasets='["set-24x24-L", "set-24x24-RGB", "set-48x48-RGB"]'
export FIDLE_OVERRIDE_GTSRB5_models='{"v1":"get_model_v1", "v2":"get_model_v2", "v3":"get_model_v3"}'
export FIDLE_OVERRIDE_GTSRB5_batch_size=64
export FIDLE_OVERRIDE_GTSRB5_epochs=5
export FIDLE_OVERRIDE_GTSRB5_scale=0.01
export FIDLE_OVERRIDE_GTSRB5_scalewith_datagen=False

# ==================================================================

echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'
echo "Notebook dir  : $NOTEBOOK_DIR"
echo "Script        : $SCRIPT_IPY"
echo "Environment   : $MODULE_ENV"
echo '------------------------------------------------------------'
env | grep FIDLE_OVERRIDE | awk 'BEGIN { FS = "=" } ; { printf("%-35s : %s\n",$1,$2) }'
echo '------------------------------------------------------------'

source /applis/environments/cuda_env.sh dahu 10.0
source /applis/environments/conda.sh
#
conda activate "$CONDA_ENV"

# ---- Run it...
#
cd $NOTEBOOK_DIR

ipython "$SCRIPT_IPY"

echo 'Done.'
