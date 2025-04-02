# Experiments

This directory contains scripts for dataset building, model training, and evaluation. To reproduce the experiments in the paper, follow these steps:

> [!NOTE]
> The user is recommended to download the dataset and preprocessed the data by following the data section in our [manuscript](https://arxiv.org/abs/2410.19343). However, to ease the trial, we provide script to generate the `raw_waveforms.h5` using STEAD dataset [link](https://github.com/smousavi05/STEAD) in `create_dataset_from_STEAD.py`. This example using STEAD dataset will *NOT* reproduce the results we provided in our manuscript.

### Installation

To set up the environment and install all dependencies follow the steps below.

1. First, download the `tqdne` code. There are two ways:

   a) **Recommended**: Download the latest [release](https://github.com/highfem/tqdne/tags) if you do not require commit history. Releases have been tested and reproduced by us and partners.

   b) Alternatively, clone the repository using:

      ```bash
      git clone (--depth 1) https://github.com/highfem/tqdne.git
      ```

      Omit `--depth 1` if you want to access the full commit history.

2. Second, create and activate a `conda` environment. Again, there are multiple options:

   a) If you prefer to create an environment in `conda`'s default path, use:

      ```bash
      conda env create -f envs/environment.yaml
      conda activate tqdne
      ```

   If conda is not installed, download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

   b) If you prefer to install the environment in a custom path, e.g., in cluster environments, run:

      ```bash
      conda env create -f envs/environment.yaml -p <PATH>
      conda activate <PATH>
      ```

      Replace `<PATH>` with your desired installation directory.


## Usage

To make running experiments as easy as possible we expect the user to adopt the following folder structure. We refer to the base folder as `workdir` which will be used to automatically store all results. The structure is as follows:

```shell
workdir/
   /data/
   /data/preprocessed_waveforms.h5
   /data/raw_waveforms.h5
   /evaluation/
   /figures/
   /outputs/
   /outputs/Autoencoder-32x96x4-LogSpectrogram
   /outputs/Latent-EDM-LogSpectrogram
```

To create each file, follow the steps below.

### Build the Raw Dataset

Download the STEAD Dataset, extract the `*.zip` file in this directory, and then simply run `create_dataset_from_STEAD.py`. It will generate

```shell
workdir/
   /data/raw_waveforms.h5
```

Note that, some of the parameters (e.g., the length of the waveforms, data sampling rate, vs30 values, and starting time of the waveforms) are hardcoded, therefore, please adjust accordingly.

### Build the Dataset

Run `build_dataset.py` to create the cleaned dataset

```shell
workdir/
   /data/preprocessed_waveforms.h5
```

from the raw `raw_waveforms.h5` file.

### Train the Classifier

Run `train_classifier.py` to train a classifier predicting the earthquake distance-magnitude bin. This classifier will be used to evaluate the generated data. Model checkpoints will be saved as

```shell
workdir/
   /outputs/Classifier-LogSpectrogram
```

### Latent Diffusion

Train the latent diffusion model using the [EDM diffusion framework](https://arxiv.org/abs/2206.00364):

1. Run `train_autoencoder.py` to train the autoencoder, the first stage of the latent diffusion model.
2. Run `train_latent_edm.py` to train the diffusion model, the second stage of the latent diffusion model.

This will create

```shell
workdir/
   /outputs/Autoencoder-32x96x4-LogSpectrogram
   /outputs/Latent-EDM-LogSpectrogram
```

**Make sure to create a soft link `best.ckpt` in `/outputs/Autoencoder-32x96x4-LogSpectrogram` such that the best checkpoint will be used for training the latent EDM.**

### Ablation Study

Conduct the following ablation studies:
1. **(No Latent) Diffusion:** Run `train_edm.py` to train the diffusion model without the autoencoder.
2. **1D Diffusion:** Run `train_1d_edm.py` to train the diffusion model generating the signal in the time domain, instead of the log-spectrogram.
2. **Latent 1D Diffusion:** Run `train_1d_autoencoder.py` followed by `train_1d_latent_edm.py` to train the latent diffusion model generating the signal in the time domain.

This will create

```shell
workdir/
   /outputs/Autoencoder-1024x16-MovingAvg
   /outputs/Latent-EDM-MovingAvg
   /outputs/EDM-LogSpectrogram
   /outputs/EDM-MovingAvg
```

### Evaluation

Run `evaluate.py` to generate synthetic seismograms conditioned on the parameters of real seismograms and evaluate them using the classifier trained in step 2. The model and classifier checkpoints and the dataset split (train, validation, or full) must be specified. Waveforms using the conditional features of the corresponding dataset split will be saved in a `.h5` file in a specified subfolder in the `outputs` directory, along with the real waveforms and classifier predictions. This file can be read by the `evaluate.ipynb` notebook to compute metrics and generate figures as presented in the paper. Check the script documentation for usage details. Example call:

```shell
python evaluate.py --split "test" --batch_size 32
```

### Generation

Run `generate.py` with a model checkpoint as an argument to generate synthetic seismograms. The generated data will be saved as a `.h5` file. Check the script documentation for usage details. Example call:

```shell
python generate.py \
    --hypocentral_distance 10.0 \
    --is_shallow_crustal 1 \
    --magnitude 5.5 \
    --vs30 760 \
    --num_samples 100 \
    --outfile workdir/generated_waveforms.h5 \
    --batch_size 32
```
