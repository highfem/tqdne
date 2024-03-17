import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "unet"
    config.rng_seq_key = 23

    config.model = new_dict(
        diffusion_model=new_dict(
            n_diffusions=100,
            solver_n_steps=10,
            solver_order=1,
            use_prior_loss=False,
            use_likelihood_loss=False
        ),
        score_model=new_dict(
            embedding_dim=256,
            use_linear_embedding=False,
            hidden_dims=[256],
            dropout_rate=0.1,
            activation=jax.nn.gelu
        ),
        noise_schedule=new_dict(
            name='linear',
            b_min=1e-04,
            b_max=0.02,
            Tmax=1.0,
        )
    )

    config.training = new_dict(
        n_iter=2000,
        early_stopping_patience=10,
        early_stopping_delta=10,
        n_mc_samples=5,
        optimizer=new_dict(
            name="adamw",
            learning_rate=0.001,
            b1=0.9,
            b2=0.999,
            weight_decay=0.00001,
        )
    )

    config.sampling = new_dict(
        sample_size=20000,
    )

    return config