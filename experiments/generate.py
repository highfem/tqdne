"""Generate waveforms using the latent EDM model.

By default, the script generates a waveform for every sample in the `gm0.h5` dataset using the corresponding conditional features.

Alternatively, a number of waveforms can be generated for a given hypocentral distance, shallow crustal flag, magnitude, and vs30 values provided as arguments. The same can be done for a set of parameters given in a CSV file, where each line should have the following format:
```
hypocentral_distance, is_shallow_crustal, magnitude, vs30, num_samples
```
where `num_samples` is the number of samples to generate for the given parameters.

The generated waveforms along with the corresponding conditional features are saved in an HDF5 file with the given name in the outputs directory.
"""

import argparse

import h5py
import numpy as np
import torch as th
from tqdm import tqdm

import tqdne.config as conf
from tqdne.autoencoder import LithningAutoencoder
from tqdne.edm import LightningEDM


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
        with open(csv, "r") as f:
            for line in f:
                args = line.split(",")
                hypercentral_distances += [float(args[0])] * int(args[4])
                is_shallow_crustals += [int(args[1])] * int(args[4])
                magnitudes += [float(args[2])] * int(args[4])
                vs30s += [float(args[3])] * int(args[4])

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
    hypercentral_distances = (
        np.array(hypercentral_distances) - dataset_hypocentral_distances.mean()
    ) / dataset_hypocentral_distances.std()
    is_shallow_crustals = (
        np.array(is_shallow_crustals) - dataset_is_shallow_crustals.mean()
    ) / dataset_is_shallow_crustals.std()
    magnitudes = (np.array(magnitudes) - dataset_magnitudes.mean()) / dataset_magnitudes.std()
    vs30s = (np.array(vs30s) - dataset_vs30s.mean()) / dataset_vs30s.std()

    cond = np.stack([hypercentral_distances, is_shallow_crustals, magnitudes, vs30s], axis=1)

    print("Loading model...")

    device = "cuda" if th.cuda.is_available() else "cpu"
    if autoencoder_checkpoint is not None:
        autoencoder = (
            LithningAutoencoder.load_from_checkpoint(config.outputdir / autoencoder_checkpoint)
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

    print(f"Generating waveforms using {device}...")

    with h5py.File(config.outputdir / output, "w") as f:
        f.create_dataset("hypocentral_distance", data=hypercentral_distances)
        f.create_dataset("is_shallow_crustal", data=is_shallow_crustals)
        f.create_dataset("magnitude", data=magnitudes)
        f.create_dataset("vs30", data=vs30s)

        waveforms = f.create_dataset("waveforms", (len(cond), 3, config.t))

        for i in tqdm(range(0, len(cond), batch_size)):
            cond_batch = cond[i : i + batch_size]
            shape = [len(cond_batch), 3, 128, 128]  # specific to latent EDM
            with th.no_grad():
                sample = edm.sample(
                    shape, cond=th.tensor(cond_batch, device=device, dtype=th.float32)
                )
            waveforms[i : i + batch_size] = config.representation.invert_representation(sample)


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
        help="output file name with generated waveforms",
    )
    parser.add_argument(
        "--config", type=str, default="LatentSpectrogramConfig", help="Config class"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--edm_checkpoint",
        type=str,
        default="Latent-EDM-LogSpectrogram/0_239-val_loss=1.18e+00.ckpt",
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
