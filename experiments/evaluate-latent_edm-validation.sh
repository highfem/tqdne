#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --partition=normal
#SBATCH --account=sd28
#SBATCH --output=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out
#SBATCH --error=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out

conda activate /capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/workdir/envs/tqdne-dev

srun torchrun --nproc_per_node=4 evaluate.py \
    --split=validation \
    --workdir=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir \
    --edm_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/Latent-EDM-32x32x4-LogSpectrogram-c128-b128-gpu4-latent4/last.ckpt \
    --autoencoder_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/Autoencoder-32x32x4-LogSpectrogram-c64-b128-gpu4-latent4/last.ckpt \
    --classifier_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/Classifier-LogSpectrogram/last.ckpt
