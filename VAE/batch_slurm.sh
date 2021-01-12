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
# <!-- TITLE --> [VAE10] - SLURM batch script
# <!-- DESC --> Bash script for SLURM batch submission of VAE8 notebooks 
# <!-- AUTHOR : Jean-Luc Parouty (CNRS/SIMaP) -->
#
# Soumission :  sbatch  /(...)/fidle/VAE/batch_slurm.sh
# Suivi      :  squeue -u $USER

# ==== Job parameters ==============================================

#SBATCH --job-name="VAE"                               # nom du job
#SBATCH --ntasks=1                                     # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                                   # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=10                             # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --hint=nomultithread                           # on réserve des coeurs physiques et non logiques
#SBATCH --time=01:00:00                                # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output="VAE_%j.out"                          # nom du fichier de sortie
#SBATCH --error="VAE_%j.err"                           # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-user=Jean-Luc.Parouty@grenoble-inp.fr
#SBATCH --mail-type=ALL

# ==== Notebook parameters =========================================

MODULE_ENV="tensorflow-gpu/py3/2.4.0"
NOTEBOOK_DIR="$WORK/fidle/VAE"

# ---- VAE MNIST
#
# NOTEBOOK_SRC="01-VAE-with-MNIST.ipynb"
# FIDLE_OVERRIDE_VAE8_run_dir="./run/MNIST.$SLURM_JOB_ID"

# ---- VAE CelebA
#
NOTEBOOK_SRC="08-VAE-with-CelebA.ipynb"
#
export FIDLE_OVERRIDE_VAE8_run_dir="./run/CelebA.$SLURM_JOB_ID"
export FIDLE_OVERRIDE_VAE8_scale="1"
export FIDLE_OVERRIDE_VAE8_image_size="(128,128)"
export FIDLE_OVERRIDE_VAE8_enhanced_dir='{datasets_dir}/celeba/enhanced'
export FIDLE_OVERRIDE_VAE8_r_loss_factor="0.7"
export FIDLE_OVERRIDE_VAE8_epochs="8"

# ---- By default (no need to modify)
#
NOTEBOOK_OUT="${NOTEBOOK_SRC%.*}==${SLURM_JOB_ID}==.ipynb"

# ==================================================================

echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'
echo "Job id        : $SLURM_JOB_ID"
echo "Job name      : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo '------------------------------------------------------------'
echo "Notebook dir  : $NOTEBOOK_DIR"
echo "Notebook src  : $NOTEBOOK_SRC"
echo "Notebook out  : $NOTEBOOK_OUT"
echo "Environment   : $MODULE_ENV"
echo '------------------------------------------------------------'
env | grep FIDLE_OVERRIDE | awk 'BEGIN { FS = "=" } ; { printf("%-35s : %s\n",$1,$2) }'
echo '------------------------------------------------------------'

# ---- Module

module purge
module load "$MODULE_ENV"

# ---- Run it...

cd $NOTEBOOK_DIR

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --output "$NOTEBOOK_OUT" --execute "$NOTEBOOK_SRC"

echo 'Done.'
