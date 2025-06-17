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
    --magnitude 5.5 \
    --vs30 760 \
    --hypocentre_depth 10.0 \
    --azimuthal_gap 130 \
    --num_samples 100 \
    --outfile workdir/generated_waveforms.h5 \
    --batch_size 32
```

### Reproducing using STEAD datasets

Although the **KiK-net** and **K-NET** datasets are freely accessible, their licenses *prohibit redistribution*. Consequently, we cannot share the pre-processed data required for full reproducibility.
Instead, you can access all preprocessing scripts in **`tqdne/scripts/preprocessing`**.

For a fully reproducible example, we provide a small dataset derived from the **STEAD** repository (Mousavi *et al.*, 2019) — [https://github.com/smousavi05/STEAD](https://github.com/smousavi05/STEAD).

To reproduce the basic **HighFEM** analysis with this STEAD sample, follow the steps outlined in *Supplementary Text S3*.

#### 1. Download the STEAD dataset

```bash
python stead_download.py --local_path /absolute/path/to/STEAD
```

*Edit `local_path` inside `stead_download.py` or pass the `--local_path` flag so files are stored where you want.*

#### 2. Convert STEAD to HDF5

```bash
python create_dataset_from_STEAD.py \
  --file_name /absolute/path/to/STEAD/waveforms \
  --csv_file /absolute/path/to/STEAD/metadata.csv \
  --output_file_path /absolute/path/to/data/raw_waveforms.h5
```

*Update the script arguments (or variables inside the script) to point to the freshly downloaded STEAD files and choose an output location.*

#### 3. Build the training dataset

```bash
python build_dataset.py --workdir /absolute/path/to/data
```

This command assumes the file `data/raw_waveforms.h5` is inside the specified `--workdir`.

#### 4. Train the autoencoder

```bash
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_autoencoder.py \
  --workdir experiments/workdir/stead \
  --mask \
  --maxlen 4064 \
  --nlatent 4 \
  --name latent-4
```

#### 5. Train the diffusion model

```bash
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_latent_edm.py \
  --workdir experiments/workdir/stead \
  --mask \
  --maxlen 4064 \
  --nlatent 4 \
  --modelchannels 128 \
  --batchsize 128 \
  --autoencodername latent-4 \
  --name c128-b128-latent-4
```

#### 6. Generate synthetic waveforms

```bash
torchrun --nproc_per_node=1 generate_stead.py \
  --workdir experiments/workdir/stead \
  --outfile experiments/workdir/stead/gwm_stead_v1.h5 \
  --edm_checkpoint experiments/workdir/stead/outputs/Latent-EDM-32x32x4-LogSpectrogram-c128-b128-latent-4/last.ckpt \
  --autoencoder_checkpoint experiments/workdir/stead/outputs/Autoencoder-32x32x4-LogSpectrogram-latent-4/last.ckpt
```

#### 7. Visualize residuals

Open and run **`Residual_plot_stead.ipynb`** to plot residuals for the generated waveforms.

---



