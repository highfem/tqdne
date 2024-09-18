# This quake does not exist

[![ci](https://github.com/highfem/tqdne/actions/workflows/ci.yaml/badge.svg)](https://github.com/highfem/tqdne/actions/workflows/ci.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2311.00474-b31b1b.svg)]()

> Generative modelling of seismic waveforms using denoising diffusion

## About

This repository contains the experimental code of the manuscript [High Resolution Seismic Waveform Generation using Denoising Diffusion](arxiv link).
It can be used to generate seismic waveforms, replicate the results from the manuscript, or for training custom models from scratch.

## Installation

You can all required dependencies and the versions that have been used in the experiments using conda.
First download the latest release of the model from [here](ttps://github.com/highfem/tqdne/releases). Then, install Python dependencies by creating a new conda environment:

```bash
conda env create -f environment.yml -p <PATH>
```

where <PATH> is a user defined path.

## Experiments

All experimental code can be found in `experiments` (which we, e.g., used to train the generative models in [1]).

## Sampling waveforms

You can generate your own waveforms using the scripts in `scripts`.

## Acknowledgements

Some Python code has been adopted from the following sources:

- EDM: https://github.com/NVlabs/edm
- Consistency models: https://github.com/openai/consistency_models

## References

[1] Bergmeister, Andreas *et al.*, [High Resolution Seismic Waveform Generation using Denoising Diffusion](arxiv link), 2024
