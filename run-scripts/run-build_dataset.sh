#!/bin/bash -l
#
#SBATCH --job-name=build_dataset
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=sd28

mamba activate tqdne

python experiments/build_dataset.py