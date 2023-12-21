#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=dif-test-%j.log
#SBATCH --error=dif-test-%j.log
#SBATCH --account=sd28

module load daint-gpu
module load cray-python
# module load TensorFlow

srun poetry run python $HOME/tqdne/experiments/train_diffusion.py