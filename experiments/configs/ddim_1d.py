# https://huggingface.co/docs/diffusers/api/schedulers/ddim

import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.name = "ddim"

    config.model = new_dict(
        scheduler_params=new_dict(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_train_timesteps=1000,
            prediction_type="sample",
            clip_sample=False,
            timestep_decimation_factor=10,
        ),
        net_params=new_dict(
            dims=1, 
            conv_kernel_size=5,  
            model_channels=32,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            num_heads=4,
            dropout=0.1,
            flash_attention=False,  # flash attention not tested (potentially faster),
            cond_emb_scale=0.1, 
        ),
        # extra_loss_terms=new_dict(
        #     signal_mean=new_dict(
        #         weight=10.0,
        #     )
        # )

    )

    config.optimizer_params=new_dict(
        learning_rate=3e-4,
        scheduler_name="cosine",
        lr_warmup_steps=400,
        batch_size=128,
        seed=0,
    )

    config.trainer_params = new_dict(
        accumulate_grad_batches=1,
        gradient_clip_val=1,
        precision="32-true",
        accelerator="auto",
        devices="auto",
        num_nodes=1,
        max_epochs=400,
        eval_every=2,
        log_to_wandb=True,
        num_sanity_val_steps=0,
        fast_dev_run=False,
        detect_anomaly=False,
    )

    config.data_repr = new_dict(
        name="SignalWithEnvelope",
        params=new_dict(
            env_function="moving_average",
            env_function_params=new_dict(
                scale=2
            ),
            env_transform="log",
            env_transform_params=new_dict(
                log_offset=1e-7
            ),
            scaling=new_dict(
                type="normalize",
                scalar=True,
                #dataset_stats_file="/users/abosisio/scratch/tqdne/outputs/log-env-stats_log-offset-1e-7_mov-avg-2_ds-2.pkl"
            ),
        )
    )

    # config.data_repr = new_dict(
    #     name="Signal",
    #     params=new_dict(
    #         scaling=new_dict(
    #             type="normalize",
    #             scalar=True
    #         ),
    #     )
    # )

    # -1: All channels
    config.metrics = new_dict(
        psd=-1, 
        logenv=-1,
        mean=-1,
    )

    # -1: All channels
    config.plots = new_dict(
        sample=-1,
        psd=-1,
        logenv=-1,
        debug=-1,
        # bin=new_dict(
        #      num_mag_bins=4,
        #      num_dist_bins=4,
        #      metrics="all",
        # )
    )

    return config