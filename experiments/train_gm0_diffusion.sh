#!/bin/bash -l
#
#SBATCH --job-name=train_ddim_cond
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=sd28

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate tqdne

python experiments/train_gm0_diffusion.py --use_last_checkpoint --config=experiments/configs/ddim_2d.py --downsampling_factor=2 --config.data_repr.params.library=librosa --config.data_repr.params.device=cpu --config.optimizer_params.batch_size=150 --train_datapath=datasets/data_train.h5 --test_datapath=datasets/data_test.h5