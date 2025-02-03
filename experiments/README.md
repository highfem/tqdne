# Experiments

This directory contains scripts for dataset building, model training, and evaluation. To reproduce the experiments in the paper, follow these steps:

### NOTE:

The user is reccommended to download the dataset and preprocessed the data by following the data section in our [manuscript](https://arxiv.org/abs/2410.19343). However, to ease the trial, we provide script to generate the `raw_waveforms.h5` using STEAD dataset [link](https://github.com/smousavi05/STEAD) in `create_dataset_from_STEAD.py`. This example using STEAD dataset will *NOT* reproduce the results we provided in our manuscript.

### Build the Raw Dataset

Download the STEAD Dataset, extract the `*.zip` file in this directory, and then simply run `create_dataset_from_STEAD.py`. It will generate `raw_waveforms.h5` file. Note that, some of the parameters (e.g., the length of the waveforms, data sampling rate, vs30 values, and starting time of the waveforms) are hardcoded, therefore, please adjust accordingly. 

### Build the Dataset

Run `build_dataset.py` to create the cleaned `preprocessed_waveforms.h5` dataset from the raw `raw_waveforms.h5` file.

### Train the Classifier

Run `train_classifier.py` to train a classifier predicting the earthquake distance-magnitude bin. This classifier will be used to evaluate the generated data. Model checkpoints will be saved in the `outputs/Classifier-LogSpectrogram` folder.

### Latent Diffusion

Train the latent diffusion model using the [EDM diffusion framework](https://arxiv.org/abs/2206.00364):

1. Run `train_autoencoder.py` to train the autoencoder, the first stage of the latent diffusion model. Model checkpoints will be saved in the `outputs/Autoencoder-LogSpectrogram-xxx` folder, where `xxx` is the size of the latent space.
2. Run `train_latent_edm.py` to train the diffusion model, the second stage of the latent diffusion model.

Make sure to specify the right path to the autoencoder checkpoint in the script.

### Ablation Study

Conduct the following ablation studies:
1. **(No Latent) Diffusion:** Run `train_edm.py` to train the diffusion model without the autoencoder.
2. **1D Diffusion:** Run `train_1d_edm.py` to train the diffusion model generating the signal in the time domain, instead of the log-spectrogram.
2. **Latent 1D Diffusion:** Run `train_1d_autoencoder.py` followed by `train_1d_latent_edm.py` to train the latent diffusion model generating the signal in the time domain.

### Generation

Run `generate.py` with a model checkpoint as an argument to generate synthetic seismograms. The generated data will be saved as a `.h5` file in a specified subfolder in the `outputs` directory. Check the script documentation for usage details.
Example call:

```bash
python generate.py --hypocentral_distance 10.0 --is_shallow_crustal 1 --magnitude 5.5 --vs30 760 --num_samples 100 --output "generated_waveforms.h5" --batch_size 32
```

### Evaluation

Run `evaluate.py` to generate synthetic seismograms conditioned on the parameters of real seismograms and evaluate them using the classifier trained in step 2. The model and classifier checkpoints and the dataset split (train, validation, or full) must be specified. Waveforms using the conditional features of the corresponding dataset split will be saved in a `.h5` file in a specified subfolder in the `outputs` directory, along with the real waveforms and classifier predictions. This file can be read by the `evaluate.ipynb` notebook to compute metrics and generate figures as presented in the paper. Check the script documentation for usage details.
Example call:
```bash
python evaluate.py --split "test" --batch_size 32
```
