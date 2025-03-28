#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=00:29:59
#SBATCH --partition=debug
#SBATCH --account=sd28
#SBATCH --output=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out
#SBATCH --error=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out

conda activate /capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/workdir/envs/tqdne-dev

srun torchrun --nproc_per_node=1 generate.py \
    --workdir=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir \
    --outfile=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/generated.h5 \
    --edm_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/Latent-EDM-LogSpectrogram/best.ckpt \
    --autoencoder_checkpoint=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/outputs/Autoencoder-32x96x4-LogSpectrogram/best.ckpt
