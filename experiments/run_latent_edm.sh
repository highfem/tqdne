#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=00:29:59
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out
#SBATCH --error=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir/slurm/%j.out

conda activate /capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/workdir/envs/tqdne-dev

srun python train_latent_edm.py \
    --workdir=/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/src/tqdne/experiments/workdir