def get_1d_autoencoder_configs(config):
    base_config = {
        "model_channels": 64,
        "channel_mult": (1, 2, 4),
        "attention_resolutions": (),
        "num_res_blocks": 2,
        "dims": 1,
        "conv_kernel_size": 5,
        "dropout": 0.1,
    }
    encoder_config = base_config | {
        "in_channels": config.channels,
        "out_channels": config.latent_channels * 2,
    }
    decoder_config = base_config | {
        "in_channels": config.latent_channels,
        "out_channels": config.channels,
    }
    return encoder_config, decoder_config


def get_1d_unet_config(config, in_channels, out_channels):
    unet_config = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "cond_features": len(config.features_keys),
        "dims": 1,
        "conv_kernel_size": 5,
        "model_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "attention_resolutions": (8,),
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "flash_attention": False,  # flash attention not tested (potentially faster)
    }
    return unet_config


def get_2d_autoencoder_configs(config):
    base_config = {
        "model_channels": 64,
        "channel_mult": (1, 2, 4),
        "attention_resolutions": (),
        "num_res_blocks": 2,
        "dims": 2,
        "conv_kernel_size": 3,
        "dropout": 0.1,
    }
    encoder_config = base_config | {
        "in_channels": config.channels,
        "out_channels": config.latent_channels * 2,
    }
    decoder_config = base_config | {
        "in_channels": config.latent_channels,
        "out_channels": config.channels,
    }
    return encoder_config, decoder_config


def get_2d_unet_config(
    config, in_channels, out_channels, model_channels=128, use_causal_mask=False
):
    unet_config = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "cond_features": len(config.features_keys),
        "dims": 2,
        "conv_kernel_size": 3,
        "model_channels": model_channels,
        "channel_mult": (1, 2, 4, 4),
        "attention_resolutions": (8,),
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "flash_attention": False,
        "use_causal_mask": use_causal_mask,
    }
    return unet_config
