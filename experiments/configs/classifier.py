# https://huggingface.co/docs/diffusers/api/schedulers/ddim

import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.name = "classifier"

    config.bins = new_dict(
            mag = [(4.5, 4.8), (4.8, 5), (5, 5.5), (5.5, 6.5), (6.5, 9.1)],
            dist = [(0, 50), (50, 100), (100, 150), (150, 200)]
    )

    config.model = new_dict(
        net_params=new_dict(
            dims=1, 
            conv_kernel_size=5,  # might want to change to 5
            model_channels=32,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            num_heads=4,
            dropout=0.2,
            flash_attention=False,  # flash attention not tested (potentially faster),
        ),
    )

    config.optimizer_params=new_dict(
        learning_rate=1e-3,
        batch_size=256,
        seed=0,
    )

    config.trainer_params = new_dict(
        accumulate_grad_batches=1,
        gradient_clip_val=1,
        precision="32-true",
        accelerator="auto",
        devices="auto",
        num_nodes=1,
        eval_every=2,
        early_stopping_patience=30,
        log_to_wandb=True,
        num_sanity_val_steps=0,
        fast_dev_run=False,
        detect_anomaly=False,
    )

    config.data_repr = new_dict(
        name="LogSpectrogram",
        params=new_dict(
            stft_channels = 128,
            hop_size = 32,
        )
    )

    return config