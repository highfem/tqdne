#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=11:59:59
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out
#SBATCH --error=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out

conda activate /capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/workdir/envs/tqdne-dev

srun python train_autoencoder.py \
    --workdir=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir