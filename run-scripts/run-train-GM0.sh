#!/bin/bash -l
#
#SBATCH --job-name=train_ddim_cond
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=sd28


export HDF5_USE_FILE_LOCKING=FALSE

mamba activate tqdne

python experiments/train_gm0_diffusion.py --config=tqdne/configs/ddim.py --downsampling_factor=2 --train_datapath=datasets/GM0-dataset-split/data_train.h5 --test_datapath=datasets/GM0-dataset-split/data_test.h5