# https://huggingface.co/docs/diffusers/api/schedulers/ddpm

import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.name = "ddpm"
    config.seed = 0 

    config.model = new_dict(
        scheduler_params=new_dict(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_train_timesteps=1000,
            prediction_type="sample",
            clip_sample=False,
        ),
        net_params=new_dict(
            cond_features=5, 
            dims=1,
            conv_kernel_size=3,  # might want to change to 5
            model_channels=32,
            num_res_blocks=2,
            num_heads=4,
            dropout=0.2,
            flash_attention=False,  # flash attention not tested (potentially faster)
        ),
        optimizer_params=new_dict(
            learning_rate=1e-4,
            lr_warmup_steps=500,
            n_train=1000,
            batch_size=32,
            max_epochs=100,
            seed=0,
        ),
    )

    config.trainer_params = new_dict(
        precision=32,
        accelerator="auto",
        devices="1",
        num_nodes=1,
    )

    config.data_repr = new_dict(
        name="SignalWithEnvelope",
        params=new_dict(
            env_function="hilbert",
            env_transform="log",
            env_transform_params=new_dict(
                log_offset=1e-5,
            ),
        )
    )


    # TODO: not sure if it is the best way to do it
    # -1: means all channels
    config.metrics = new_dict(
        psd=-1, 
    )

    # TODO: not sure if it is the best way to do it
    config.plots = new_dict(
        sample=-1,
        psd=-1,
        bin=new_dict(
            num_mag_bins=4,
            num_dist_bins=4,
            metrics="all",
        )
    )

    return config