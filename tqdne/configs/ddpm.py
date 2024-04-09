# https://huggingface.co/docs/diffusers/api/schedulers/ddpm

import ml_collections

def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.name = "ddpm"

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
            dims=1,
            conv_kernel_size=3,  # might want to change to 5
            model_channels=32,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            num_heads=4,
            dropout=0.1,
            flash_attention=False,  # flash attention not tested (potentially faster),
            cond_emb_scale=None, 
        ),
    )

    config.optimizer_params=new_dict(
        learning_rate=1e-3,
        scheduler_name="cosine",
        lr_warmup_steps=5000,
        batch_size=64,
        seed=0,
    )

    config.trainer_params = new_dict(
        accumulate_grad_batches=1,
        gradient_clip_val=1,
        precision="32-true",
        accelerator="auto",
        devices="auto",
        num_nodes=1,
        max_epochs=100,
        eval_every=2,
        log_to_wandb=True,
        num_sanity_val_steps=0,
        fast_dev_run=False,
        detect_anomaly=False,
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
            scaling=new_dict(
                type="normalize",
                scalar=True
            ),
        )
    )


    # TODO: not sure if it is the best way to do it
    # -1: means all channels
    config.metrics = new_dict(
        psd=-1,
        logenv=-1,
    )

    # TODO: not sure if it is the best way to do it
    config.plots = new_dict(
        sample=-1,
        psd=-1,
        logenv=-1,
        debug=-1,
        bin=new_dict(
            num_mag_bins=4,
            num_dist_bins=4,
            metrics="all",
        )
    )

    return config