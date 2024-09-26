#!/bin/bash -l
#
#SBATCH --job-name=train_classifier
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=sd28


export HDF5_USE_FILE_LOCKING=FALSE

mamba activate tqdne

python experiments/train_eval_classifier.py --config=configs/classifier.py --downsampling_factor=2 --config.optimizer_params.batch_size=128 --train_datapath=datasets/data_train.h5 --test_datapath=datasets/data_test.h5