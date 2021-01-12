#!/bin/bash
# -----------------------------------------------
#         _           _       _
#        | |__   __ _| |_ ___| |__
#        | '_ \ / _` | __/ __| '_ \
#        | |_) | (_| | || (__| | | |
#        |_.__/ \__,_|\__\___|_| |_|
#                              Fidle at IDRIS
# -----------------------------------------------
#
# <!-- TITLE --> [GTSRB11] - SLURM batch script
# <!-- DESC --> Bash script for a Slurm batch submission of an ipython code
# <!-- AUTHOR : Jean-Luc Parouty (CNRS/SIMaP) -->
#
# Soumission :  sbatch  /(...)/fidle/GTSRB/batch_slurm.sh
# Suivi      :  squeue -u $USER

# ==== Job parameters ==============================================

#SBATCH --job-name="GTSRB"                             # nom du job
#SBATCH --ntasks=1                                     # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                                   # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=10                             # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --hint=nomultithread                           # on réserve des coeurs physiques et non logiques
#SBATCH --time=01:00:00                                # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output="GTSRB_%j.out"                        # nom du fichier de sortie
#SBATCH --error="GTSRB_%j.err"                         # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-user=Jean-Luc.Parouty@grenoble-inp.fr
#SBATCH --mail-type=ALL

# ==== Notebook parameters =========================================

MODULE_ENV="tensorflow-gpu/py3/2.4.0"
NOTEBOOK_DIR="$WORK/fidle/GTSRB"

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
echo "Job id        : $SLURM_JOB_ID"
echo "Job name      : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo '------------------------------------------------------------'
echo "Notebook dir  : $NOTEBOOK_DIR"
echo "Script        : $SCRIPT_IPY"
echo "Environment   : $MODULE_ENV"
echo '------------------------------------------------------------'
env | grep FIDLE_OVERRIDE | awk 'BEGIN { FS = "=" } ; { printf("%-35s : %s\n",$1,$2) }'
echo '------------------------------------------------------------'

# ---- Module

module purge
module load "$MODULE_ENV"

# ---- Run it...

cd $NOTEBOOK_DIR

ipython "$SCRIPT_IPY"

echo 'Done.'