#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=wgan64-test-%j.log
#SBATCH --error=wgan64-test-%j.log
#SBATCH --account=sd28

module load daint-gpu
module load cray-python
# module load TensorFlow

srun poetry run python $HOME/tqdne/experiments/test_wgan.py