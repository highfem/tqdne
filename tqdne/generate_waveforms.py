#!/usr/bin/env python

"""Generate waveforms using the trained EDM model."""

import argparse
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import torch as th
from tqdm import tqdm

import tqdne
from tqdne.autoencoder import LightningAutoencoder
from tqdne.edm import LightningEDM
from tqdne.representation import LogSpectrogram
from tqdne.utils import get_device


@dataclass
class LatentSpectrogramConfig:
    features_keys: tuple[str] = (
        "hypocentral_distance",
        "magnitude",
        "vs30",
        "hypocentre_depth",
        "azimuthal_gap",
    )
    channels: int = 3
    fs: int = 100
    stft_channels: int = 256
    hop_size: int = 32
    representation = LogSpectrogram(stft_channels=stft_channels, hop_size=hop_size)
    t: int = 4096 - hop_size
    latent_channels: int = 4
    kl_weight: float = 1e-6


def download_checkpoints():
    response = requests.get("https://zenodo.org/records/15687691/files/tqdne-0.2.0.zip", timeout=60)
    with open("downloaded_file.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
        zip_ref.extractall(".data")
    os.remove("downloaded_file.zip")


def get_checkpoints(edm_checkpoint, autoencoder_checkpoint):
    if edm_checkpoint is None and autoencoder_checkpoint is None:
        print("downloading checkpoints from zenodo...")
        if not os.path.exists(".data"):
            download_checkpoints()
        edm_checkpoint = ".data/tqdne-0.2.0/weights/edm.ckpt"
        autoencoder_checkpoint = ".data/tqdne-0.2.0/weights/autoencoder.ckpt"
    elif edm_checkpoint is None or autoencoder_checkpoint is None:
        raise ValueError("Either both or none of the checkpoints must be provided.")
    else:
        print("using provided checkpoints")
    return edm_checkpoint, autoencoder_checkpoint


@th.no_grad()
def generate(
    hypocentral_distance,
    magnitude,
    vs30,
    hypocentre_depth,
    azimuthal_gap,
    num_samples,
    csv,
    outfile,
    batch_size,
    edm_checkpoint,
    autoencoder_checkpoint,
):
    edm_checkpoint, autoencoder_checkpoint = get_checkpoints(edm_checkpoint, autoencoder_checkpoint)
    if csv:
        print("using csv data")
        df = pd.read_csv(csv)
        df = df.loc[df.index.repeat(df.num_samples)]
        hypocentral_distances = df.hypocentral_distance.to_list()
        hypocentral_distances = np.array(hypocentral_distances) * 1e-3
        magnitudes = df.magnitude.to_list()
        vs30s = df.vs30.to_list()
        hypocentre_depths = df.hypocentre_depth.to_list()
        azimuthal_gaps = df.azimuthal_gap.to_list()
    elif np.all(
        [
            c is not None
            for c in [
                hypocentral_distance,
                magnitude,
                vs30,
                hypocentre_depth,
                magnitude,
                num_samples,
            ]
        ]
    ):
        print("using command line input data")
        hypocentral_distances = [hypocentral_distance] * num_samples * 1e-3
        magnitudes = [magnitude] * num_samples
        vs30s = [vs30] * num_samples
        hypocentre_depths = [hypocentre_depth] * num_samples
        azimuthal_gaps = [azimuthal_gap] * num_samples
    else:
        raise ValueError("provide either a CSV or a full parameter set")

    print("loading model...")
    config = LatentSpectrogramConfig()
    device = get_device()
    autoencoder = None
    if autoencoder_checkpoint is not None:
        autoencoder = (
            LightningAutoencoder.load_from_checkpoint(autoencoder_checkpoint).to(device).eval()
        )
    edm = (
        LightningEDM.load_from_checkpoint(edm_checkpoint, autoencoder=autoencoder).to(device).eval()
    )

    # these are the summary statistics of the conditional features of the entire data set
    # we need them to normalize the inputs
    summary_statistics = np.array(
        [
            [0.10196523161124325, 0.040784282611932066],
            [4.730266447282116, 0.6278049031313433],
            [371.9033679970052, 214.68149233229866],
            [39.06279937237003, 22.487259252475628],
            [129.88750553380498, 89.67082312013503],
        ]
    )

    # normalize features
    hypocentral_distances_norm = (
        np.array(hypocentral_distances) - summary_statistics[0, 0]
    ) / summary_statistics[0, 1]
    magnitudes_norm = (np.array(magnitudes) - summary_statistics[1, 0]) / summary_statistics[1, 1]
    vs30s_norm = (np.array(vs30s) - summary_statistics[2, 0]) / summary_statistics[2, 1]
    hypocentre_depths_norm = (
        np.array(hypocentre_depths) - summary_statistics[3, 0]
    ) / summary_statistics[3, 1]
    azimuthal_gaps_norm = (
        np.array(azimuthal_gaps) - summary_statistics[4, 0]
    ) / summary_statistics[4, 1]
    cond = np.stack(
        [
            hypocentral_distances_norm,
            magnitudes_norm,
            vs30s_norm,
            hypocentre_depths_norm,
            azimuthal_gaps_norm,
        ],
        axis=1,
    )

    print("loading models...")
    device = get_device()
    autoencoder = None
    if autoencoder_checkpoint is not None:
        autoencoder_checkpoint = Path(autoencoder_checkpoint)
        autoencoder = (
            LightningAutoencoder.load_from_checkpoint(autoencoder_checkpoint).to(device).eval()
        )

    th.serialization.add_safe_globals([tqdne.edm.EDM])
    edm_checkpoint = Path(edm_checkpoint)
    edm = (
        LightningEDM.load_from_checkpoint(edm_checkpoint, autoencoder=autoencoder).to(device).eval()
    )

    print(f"generating waveforms using {device}...")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("hypocentral_distance", data=np.array(hypocentral_distances) * 1e3)
        f.create_dataset("magnitude", data=np.array(magnitudes))
        f.create_dataset("vs30s", data=np.array(vs30s))
        f.create_dataset("hypocentre_depth", data=np.array(hypocentre_depths))
        f.create_dataset("azimuthal_gap", data=np.array(azimuthal_gaps))

        waveforms = f.create_dataset("waveforms", (len(cond), 3, config.t))

        for i in tqdm(range(0, len(cond), batch_size)):
            cond_batch = cond[i : i + batch_size]
            shape = [len(cond_batch), 3, 128, 128]
            with th.no_grad():
                sample = edm.sample(
                    shape, cond=th.tensor(cond_batch, device=device, dtype=th.float32)
                )
            waveforms[i : i + batch_size] = config.representation.invert_representation(sample)
    print("done!")


def main():
    desc = """Generate waveforms using the trained EDM model.

By default, the script generates a waveform for every sample in the test set of
`preprocessed_waveforms.h5` dataset using the corresponding conditional features.
Alternatively, a number of waveforms can be generated for a given set of conditioning
variables values provided as arguments. The same can be done for a set of parameters
given in a CSV file, where each line should have the following format:
```
hypocentral_distance,magnitude,vs30,hypocentre_depth,azimuthal_gap,num_samples
```
where `num_samples` is the number of samples to generate for the given parameters.
The generated waveforms along with the corresponding conditional features
are saved in an HDF5 file with the given name in the outputs directory.
"""
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--hypocentral_distance (km)", type=float, default=None)
    parser.add_argument("--magnitude (unitless)", type=float, default=None)
    parser.add_argument("--vs30 (m/s)", type=float, default=None)
    parser.add_argument("--hypocentre_depth (km)", type=float, default=None)
    parser.add_argument("--azimuthal_gap (deg)", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--csv", type=str, default=None, help="csv file with args")
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output file name with generated waveforms",
    )
    parser.add_argument(
        "--edm_checkpoint",
        type=str,
        required=False,
        help="EDM checkpoint. If not provided, will automatically download from Zenodo.",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        required=False,
        help="Autoencoder checkpoint. If not provided, will automatically download from Zenodo.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device.")
    args = parser.parse_args()

    generate(
        args.hypocentral_distance,
        args.magnitude,
        args.vs30,
        args.hypocentre_depth,
        args.azimuthal_gap,
        args.num_samples,
        args.csv,
        args.outfile,
        args.batch_size,
        args.edm_checkpoint,
        args.autoencoder_checkpoint,
    )
