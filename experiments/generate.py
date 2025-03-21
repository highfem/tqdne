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
    is_shallow_crustal,
    magnitude,
    vs30,
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
    is_shallow_crustal : int
        Shallow crustal flag (0 or 1).
    magnitude : float
        Earthquake magnitude.
    vs30 : float
        Vs30 value.
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
        dataset_is_shallow_crustals = f["is_shallow_crustal"][:]
        dataset_magnitudes = f["magnitude"][:]
        dataset_vs30s = f["vs30"][:]

    if csv:
        hypercentral_distances = []
        is_shallow_crustals = []
        magnitudes = []
        vs30s = []
        with open(csv) as f:
            f.readline()  # skip header
            for line in f:
                args = line.strip().split(",")
                num_samples = int(args[4])
                hypercentral_distances += [float(args[0])] * num_samples
                is_shallow_crustals += [float(bool(args[1]))] * num_samples
                magnitudes += [float(args[2])] * num_samples
                vs30s += [float(args[3])] * num_samples

    elif (
        hypocentral_distance is not None
        and is_shallow_crustal is not None
        and magnitude is not None
        and vs30 is not None
        and num_samples is not None
    ):
        hypercentral_distances = [hypocentral_distance] * num_samples
        is_shallow_crustals = [is_shallow_crustal] * num_samples
        magnitudes = [magnitude] * num_samples
        vs30s = [vs30] * num_samples
    else:
        hypercentral_distances = dataset_hypocentral_distances
        is_shallow_crustals = dataset_is_shallow_crustals
        magnitudes = dataset_magnitudes
        vs30s = dataset_vs30s

    # normalize features
    hypercentral_distances_norm = (
        np.array(hypercentral_distances) - dataset_hypocentral_distances.mean()
    ) / dataset_hypocentral_distances.std()
    is_shallow_crustals_norm = (
        np.array(is_shallow_crustals) - dataset_is_shallow_crustals.mean()
    ) / dataset_is_shallow_crustals.std()
    magnitudes_norm = (np.array(magnitudes) - dataset_magnitudes.mean()) / dataset_magnitudes.std()
    vs30s_norm = (np.array(vs30s) - dataset_vs30s.mean()) / dataset_vs30s.std()

    cond = np.stack(
        [hypercentral_distances_norm, is_shallow_crustals_norm, magnitudes_norm, vs30s_norm], axis=1
    )

    print("Loading model...")

    device = get_device()
    if autoencoder_checkpoint is not None:
        autoencoder = (
            LightningAutoencoder.load_from_checkpoint(config.outputdir / autoencoder_checkpoint)
            .to(device)
            .eval()
        )
    edm = (
        LightningEDM.load_from_checkpoint(
            config.outputdir / edm_checkpoint, autoencoder=autoencoder
        )
        .to(device)
        .eval()
    )

    print("Load Dataset to infer output shape...")

    dataset = Dataset(config.datapath, config.representation, cut=config.t)
    signal_shape = dataset[0]["signal"].shape

    print(f"Generating waveforms using {device}...")

    with h5py.File(config.outputdir / output, "w") as f:
        f.create_dataset("hypocentral_distance", data=np.array(hypercentral_distances))
        f.create_dataset("is_shallow_crustal", data=np.array(is_shallow_crustals))
        f.create_dataset("magnitude", data=np.array(magnitudes))
        f.create_dataset("vs30", data=np.array(vs30s))

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
    parser.add_argument("--is_shallow_crustal", type=int, default=None)
    parser.add_argument("--magnitude", type=float, default=None)
    parser.add_argument("--vs30", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--csv", type=str, default=None, help="csv file with args")
    parser.add_argument(
        "--output",
        type=str,
        default="generated.h5",
        help="Output file name with generated waveforms",
    )
    parser.add_argument(
        "--config", type=str, default="LatentSpectrogramConfig", help="Config class"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--edm_checkpoint",
        type=str,
        default="Latent-EDM-LogSpectrogram/0_299-val_loss=1.18e+00.ckpt",
        help="EDM checkpoint",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        default="Autoencoder-32x32x4-LogSpectrogram/0_199-val_loss=1.55e-03.ckpt",
        help="Optional autoencoder checkpoint. Needed for Latent-EDM.",
    )
    args = parser.parse_args()

    config = getattr(conf, args.config)()
    if "latent" not in args.config.lower():
        args.autoencoder_checkpoint = None

    generate(
        args.hypocentral_distance,
        args.is_shallow_crustal,
        args.magnitude,
        args.vs30,
        args.num_samples,
        args.csv,
        args.output,
        config,
        args.batch_size,
        args.edm_checkpoint,
        args.autoencoder_checkpoint,
    )
