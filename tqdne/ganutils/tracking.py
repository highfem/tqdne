#!/usr/bin/env python3

import mlflow
from torchinfo import summary


def log_params_mlflow(params):
    """log_params_mlflow add config parameters to mlflow tracking. 

    :param params: argument parser instance
    :type params: ParamParser
    """
    
    mlflow.log_param("GPUs", params.gpus)
    mlflow.log_param("Model file", params.model_file)
    mlflow.log_param("Data file", params.data_file)
    mlflow.log_param("Attribute file", params.attr_file)
    mlflow.log_param("Learning rate", params.learning_rate)
    mlflow.log_param("Discriminator size", params.discriminator_size)
    mlflow.log_param("dt", params.time_delta)
    mlflow.log_param("Generator noise dimension", params.noise_dim)
    mlflow.log_param("gp lambda", params.gp_lambda)
    mlflow.log_param("Critic iterations per cycle", params.n_critic)
    mlflow.log_param("Optimizer Beta 1", params.beta1)
    mlflow.log_param("Optimizer Beta 2", params.beta2)
    mlflow.log_param("Epochs", params.epochs)
    mlflow.log_param("Batch size", params.batch_size)
    mlflow.log_param("Learning rate", params.learning_rate)
    mlflow.log_param("Sample rate", params.sample_rate)
    mlflow.log_param("Number of conditonal variable bins", params.n_cond_bins)
    mlflow.log_param("Train test split fraction", params.frac_train)


def log_model_mlflow(D, G, out_dir):
    """log_model_mlflow _summary_

    _extended_summary_

    :param D: the discriminator object
    :type D: Discriminator
    :param G: the generator object 
    :type G: Generator
    :param out_dir: artifact folder path where mlflow is logging the artifacts.
    :type out_dir: string
    """

    with open(f'{out_dir}/generator.txt', 'w') as f:
        f.write(str(summary(G)))
    mlflow.log_artifact(f'{out_dir}/generator.txt', "Generator")
    
    with open(f'{out_dir}/discriminator.txt', 'w') as f:
        f.write(str(summary(D)))
    mlflow.log_artifact(f'{out_dir}/discriminator.txt', "Discriminator")
    
    with open(f'{out_dir}/generator_state_dict.txt', 'w') as f:
        f.write(str(G.state_dict()))
    mlflow.log_artifact(f'{out_dir}/generator_state_dict.txt', "Generator state dict")

    with open(f'{out_dir}/discriminator_state_dict.txt', 'w') as f:
        f.write(str(D.state_dict()))
    mlflow.log_artifact(f'{out_dir}/discriminator_state_dict.txt', "Discriminator state dict")    
    
    mlflow.log_param("Generator_num_params", sum(p.numel() for p in G.parameters() if p.requires_grad))
    mlflow.log_param("Discriminator_num_params", sum(p.numel() for p in D.parameters() if p.requires_grad))
