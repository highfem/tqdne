"""Generate waveforms using the trained EDM model.

By default, the script generates a waveform for every sample in the `gm0.h5` dataset using the corresponding conditional features.

Alternatively, a number of waveforms can be generated for a given hypocentral distance, shallow crustal flag, magnitude, and vs30 values provided as arguments. The same can be done for a set of parameters given in a CSV file, where each line should have the following format:
```
hypocentral_distance, is_shallow_crustal, magnitude, vs30, num_samples
```
where `num_samples` is the number of samples to generate for the given parameters,
`is_shallow_crustal` is a boolean flag (0 or 1), and the rest are floating-point values.

The generated waveforms along with the corresponding conditional features are saved in an HDF5 file with the given name in the outputs directory.
"""

import argparse

import pandas as pd

import config as conf
import h5py
import numpy as np
import torch as th
from tqdm import tqdm

from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM
from tqdne.utils import get_device


@th.no_grad()
def generate(
    hypocentral_distance,
    magnitude,
    vs30,
    hypocentre_depth,
    azimuthal_gap,
    num_samples,
    csv,
    output,
    config,
    batch_size,
    edm_checkpoint,
    autoencoder_checkpoint,
):
    """Generate waveforms using the trained EDM model.

    Parameters
    ----------
    hypocentral_distance : float
        Hypocentral distance.
    magnitude : float
        Earthquake magnitude.
    vs30 : float
        Vs30 value.
    hypocentre_depth : float
        hypocentre_depth
    azimuthal_gap : float
        azimuthal_gap
    num_samples : int
        Number of samples to generate for the given parameters.
    csv : str
        CSV file with parameters.
    output : str
        Output file name with generated waveforms.
    config : str
        One of the configuration classes in `tqdne.config`.
    batch_size : int
        Batch size used for the generation.
    edm_checkpoint : str
        Saved EDM model checkpoint. Relative to the output directory.
    autoencoder_checkpoint : str
        Optional autoencoder model checkpoint. Needed for the Latent-EDM model. Relative to the output directory.
    """

    print("Prepare conditional features...")
    with h5py.File(config.datapath, "r") as f:
        dataset_hypocentral_distances = f["hypocentral_distance"][:]
        dataset_magnitudes = f["magnitude"][:]
        dataset_vs30s = f["vs30"][:]
        dataset_hypocentre_depths = f["hypocentre_depth"][:]
        dataset_azimuthal_gap = f["azimuthal_gap"][:]

    if csv:
        df = pd.read_csv(csv)
        df = df.loc[df.index.repeat(df.num_samples)]
        hypocentral_distances = df.hypocentral_distance.to_list()
        magnitudes = df.magnitude.to_list()
        vs30s = df.vs30.to_list()
        hypocentre_depths = df.hyhypocentre_depth.to_list()
        azimuthal_gaps = df.azimuthal_gap.to_list()

    elif np.all([
        c is not None
        for c in [hypocentral_distance, magnitude, vs30, hypocentre_depth, magnitude, num_samples]
    ]):
        hypocentral_distances = [hypocentral_distance] * num_samples
        magnitudes = [magnitude] * num_samples
        vs30s = [vs30] * num_samples
        hypocentre_depths = [hypocentre_depth] * num_samples
        azimuthal_gaps = [azimuthal_gap] * num_samples

    else:
        hypocentral_distances = dataset_hypocentral_distances
        magnitudes = dataset_magnitudes
        vs30s = dataset_vs30s
        hypocentre_depths = dataset_hypocentre_depths
        azimuthal_gaps = dataset_azimuthal_gap

    # normalize features
    hypocentral_distances_norm = (
        np.array(hypocentral_distances) - dataset_hypocentral_distances.mean()
    ) / dataset_hypocentral_distances.std()
    magnitudes_norm = (np.array(magnitudes) - dataset_magnitudes.mean()) / dataset_magnitudes.std()
    vs30s_norm = (np.array(vs30s) - dataset_vs30s.mean()) / dataset_vs30s.std()
    hypocentre_depths_norm = (
        np.array(hypocentre_depths) - dataset_hypocentre_depths.mean()
    ) / dataset_hypocentre_depths.std()
    azimuthal_gaps_norm = (
        np.array(azimuthal_gaps) - dataset_azimuthal_gap.mean()
    ) / dataset_azimuthal_gap.std()

    cond = np.stack([
        hypocentral_distances_norm,
        magnitudes_norm,
        vs30s_norm,
        hypocentre_depths_norm,
        azimuthal_gaps_norm
        ], axis=1
    )

    print("Loading model...")

    device = get_device()
    autoencoder = None
    if autoencoder_checkpoint is not None:
        autoencoder = (
            LightningAutoencoder.load_from_checkpoint(autoencoder_checkpoint)
            .to(device)
            .eval()
        )
    edm = (
        LightningEDM.load_from_checkpoint(edm_checkpoint, autoencoder=autoencoder)
        .to(device)
        .eval()
    )

    print("Load Dataset to infer output shape...")

    dataset = Dataset(config.datapath, config.representation, cut=config.t)
    signal_shape = dataset[0]["signal"].shape

    print(f"Generating waveforms using {device}...")
    with h5py.File(config.outputdir / output, "w") as f:
        f.create_dataset("hypocentral_distance", data=np.array(hypocentral_distances))
        f.create_dataset("magnitude", data=np.array(magnitudes))
        f.create_dataset("hypocentre_depth", data=np.array(hypocentre_depths))
        f.create_dataset("azimuthal_gap", data=np.array(azimuthal_gaps))

        waveforms = f.create_dataset("waveforms", (len(cond), 3, config.t))

        for i in tqdm(range(0, len(cond), batch_size)):
            cond_batch = cond[i : i + batch_size]
            shape = [len(cond_batch), *signal_shape]
            with th.no_grad():
                sample = edm.sample(
                    shape, cond=th.tensor(cond_batch, device=device, dtype=th.float32)
                )
            waveforms[i : i + batch_size] = config.representation.invert_representation(sample)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate waveforms")
    parser.add_argument("--hypocentral_distance", type=float, default=None)
    parser.add_argument("--magnitude", type=float, default=None)
    parser.add_argument("--vs30", type=float, default=None)
    parser.add_argument("--hypocentre_depth", type=float, default=None)
    parser.add_argument("--azimuthal_gap", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--csv", type=str, default=None, help="csv file with args")
    parser.add_argument(
        "--workdir", type=str, help="the working directory in which checkpoints and all outputs are saved to (same as used during training)"
    )
    parser.add_argument(
        "--edm_checkpoint",
        type=str,
        required=True,
        help="EDM checkpoint",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        help="Optional autoencoder checkpoint. Needed for Latent-EDM.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output file name with generated waveforms; if not given writes to workdir/outputs/generated.h5",
    )
    parser.add_argument(
        "--config", type=str, default="LatentSpectrogramConfig", help="Config class"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    config = getattr(conf, args.config)()
    if "latent" not in args.config.lower():
        args.autoencoder_checkpoint = None

    generate(
        args.hypocentral_distance,
        args.magnitude,
        args.vs30,
        args.hypocentre_depth,
        args.azimuthal_gap,
        args.num_samples,
        args.csv,
        args.output,
        config,
        args.batch_size,
        args.edm_checkpoint,
        args.autoencoder_checkpoint,
    )
