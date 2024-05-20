# https://huggingface.co/docs/diffusers/api/pipelines/consistency_models

import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "consistency-model"
    config.seed = 0 

    config.model = new_dict(
        net_params = new_dict(
            cond_features= 5,  # set to 5 if cond=True
            dims= 1,
            conv_kernel_size= 3,
            model_channels= 32,
            channel_mult= (1, 2, 4, 8),  # might want to change to (1, 2, 4, 4)
            num_res_blocks= 2,
            num_heads= 4,
            dropout= 0.2,
            flash_attention= False,  # flash attention not tested (potentially faster)
        ),
        optimizer_params=new_dict(
            lr=1e-3,
            batch_size=64,
        ),
    )

    config.trainer_params = new_dict(
        max_epochs= 100,
        precision= 32,
        accelerator= "auto",
        devices= "1",
        num_nodes= 1,
    )

    config.data_repr = new_dict(
        name="SignalWithEnvelope",
        params=new_dict(
            env_function="hilbert",
            env_function_params=new_dict(),
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
        logenv=-1,
        # bin=new_dict(
        #     num_mag_bins=4,
        #     num_dist_bins=4,
        #     metrics="all",
        # )
    )

    return config