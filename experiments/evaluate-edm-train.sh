#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --partition=normal
#SBATCH --account=sd28
#SBATCH --output=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out
#SBATCH --error=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out

conda activate /capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/workdir/envs/tqdne-dev

srun torchrun --nproc_per_node=4 evaluate.py \
    --split=train \
    --workdir=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir \
    --edm_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/EDM-128x128x4-LogSpectrogram/last.ckpt \
    --config=SpectrogramConfig \
    --classifier_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/Classifier-LogSpectrogram/last.ckpt
