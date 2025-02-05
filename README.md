# This quake does not exist

[![ci](https://github.com/highfem/tqdne/actions/workflows/ci.yml/badge.svg)](https://github.com/highfem/tqdne/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2410.19343-b31b1b.svg)](https://arxiv.org/abs/2410.19343)

> Generative modelling of seismic waveforms using denoising diffusion.
> 
> ![Generative pipeline](pipeline.jpg)

## About

This repository contains the experimental code of the manuscript [High Resolution Seismic Waveform Generation using Denoising Diffusion](https://arxiv.org/abs/2410.19343).
It can be used to generate seismic waveforms, replicate the results from the manuscript, or for training custom models from scratch.

## Installation

To set up the environment and install dependencies:

1. **Clone the repository:**

   ```bash
   git clone --depth 1 https://github.com/highfem/tqdne.git
   cd tqdne
   ```

   Ommit `--depth 1` if you want to access the full commit history.

2. **Create and activate a conda environment:**

   ```bash
   conda env create -f environment.yaml
   conda activate tqdne
   ```

   If conda is not installed, download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. (Optional) If you prefer to install the environment in a custom path, run:

   ```bash
   conda env create -f environment.yaml -p <PATH>
   conda activate <PATH>
   ```

   Replace `<PATH>` with your desired installation directory.

## Weight files

You can find the weight files for the neural networks on [Zenodo](https://zenodo.org/records/13952381), under the `weights` folder.

## Experiments

To reproduce the experiments from the manuscript, including data preprocessing, training, and evaluation, navigate to the [experiments](./experiments) folder. Refer to the corresponding README files for step-by-step guidance.

## Sampling waveforms

You can generate your own waveforms using the scripts in [scripts](./scripts). See the corresponding README files for more information.

## Acknowledgements

Some Python code has been adapted from the following repositories:

- [EDM](https://github.com/NVlabs/edm)
- [Consistency Models](https://github.com/openai/consistency_models)
