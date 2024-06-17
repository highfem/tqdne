# **T**his **Q**uake **D**oes **N**ot **E**xist

This project provides modules and scripts to train, generate, and evaluate deep learning models for creating synthetic seismic data.

## Setup

Follow these steps to get started:

1. **Clone the repository:**
    ```bash
    git clone git@github.com:highfem/tqdne.git
    cd tqdne
    ```

2. **Add the seismic waveforms file:**
    - Copy `wforms_GAN_input_v20220805.h5` to the `datasets/` folder. (This file is not included in the repository due to size constraints.)

3. **Install dependencies:**
    - Ensure you have conda installed. If not, install [miniforge](https://github.com/conda-forge/miniforge) (recommended) or any other conda implementation.
    - Create the `tqne` conda environment:
    ```bash
    conda env create -f environment.yml
    ```

4. **Activate the environment:**
    - Perform this step each time you work on the project.
    ```bash
    conda activate tqne
    ```

## Structure

The project is organized as follows:

- `tqne/`: Python package with modules for training, generating, and evaluating synthetic seismic data.
- `experiments/`: Scripts for dataset building, model training, and evaluation. Refer to the README.md in this folder for details.
- `notebooks/`: Jupyter notebooks for data exploration and visualization.
- `datasets/`: Directory for raw and processed datasets. Place `wforms_GAN_input_v20220805.h5` here before running experiments.
- `outputs/`: Directory for experiment outputs, including model checkpoints and generated data.
