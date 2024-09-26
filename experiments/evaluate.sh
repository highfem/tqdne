#!/bin/bash -l
#
#SBATCH --job-name=evaluate
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=sd28

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate tqdne

python experiments/evaluate.py --diffusion_model_checkpoint=outputs/ddim-pred:sample-1D-downsampling:2_SignalWithEnvelope-moving_average-scale:2-log-log_offset:1.0e-07-normalize-scalar:True/name=0_epoch=186-val_loss=0.14.ckpt --batch_size=512 --outputdir=1d_eval