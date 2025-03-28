"""Evaluate the trained EDM model.

This script generates waveforms using the same conditional features as the dataset.
The generated waveforms are saved along with original waveforms, conditional features, and classifier outputs in an HDF5 file.
The created file can be read by the corresponding notebook to compute metrics and create plots.
"""

import sys
from argparse import ArgumentParser

import config as conf
import torch
import torch as th
from h5py import File
from torch.utils.data import DataLoader
from tqdm import tqdm

from tqdne.autoencoder import LightningAutoencoder
from tqdne.classifier import LithningClassifier
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM
from tqdne.utils import get_device


@torch.no_grad()
def predict(
    split,
    outputdir,
    config,
    classifier_config,
    batch_size,
    edm_checkpoint,
    classifier_checkpoint,
    autoencoder_checkpoint,
):
    """Evaluate the trained EDM model.

    Parameters
    ----------
    split : str
        Dataset split (`train`, `test`, or `full`).
    outputdir : str
        Output directory for the evaluation results.
    config : Config
        One of the configuration classes in `tqdne.config`.
    classifier_config : str
        One of the configuration classes in `tqdne.config`.
    batch_size : int
        Batch size used for the generation.
    edm_checkpoint : str
        Saved EDM model checkpoint. Relative to the output directory.
    classifier_checkpoint : str
        Saved classifier model checkpoint. Relative to the output directory.
    autoencoder_checkpoint : str
        Optional autoencoder model checkpoint. Needed for the Latent-EDM model. Relative to the output directory.
    """

    print(f"Predicting {split} set...")

    dataset = Dataset(config.datapath, config.representation, cut=config.t, cond=True, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    print("Loading model...")

    device = get_device()
    autoencoder = (
        LightningAutoencoder.load_from_checkpoint(config.outputdir / autoencoder_checkpoint)
        if autoencoder_checkpoint is not None
        else None
    )
    edm = (
        LightningEDM.load_from_checkpoint(
            config.outputdir / edm_checkpoint, autoencoder=autoencoder
        )
        .to(device)
        .eval()
    )

    classifier = (
        LithningClassifier.load_from_checkpoint(config.outputdir / classifier_checkpoint)
        .to(device)
        .eval()
    )

    # generate a single batch to get the output size
    batch = next(iter(loader))
    signal_shape = batch["signal"].shape[1:]
    waveform_shape = batch["waveform"].shape[1:]
    classifier_embedding = classifier.embed(
        th.tensor(
            classifier_config.representation.get_representation(batch["waveform"]), device=device
        )
    )
    classifier_embedding_shape = classifier_embedding.shape[1:]
    classifier_pred_shape = classifier.output_layer(classifier_embedding).shape[1:]

    outputdir = config.outputdir / outputdir
    outputdir.mkdir(exist_ok=True)
    with File(outputdir / f"{split}.h5", "w") as f:
        for key in config.features_keys:
            f.create_dataset(key, data=dataset.get_feature(key))

        target_waveform = f.create_dataset(
            "target_waveform", shape=(len(dataset), *waveform_shape), dtype="f"
        )
        predicted_waveform = f.create_dataset(
            "predicted_waveform", shape=(len(dataset), *waveform_shape), dtype="f"
        )
        target_signal = f.create_dataset(
            "target_signal", shape=(len(dataset), *signal_shape), dtype="f"
        )
        predicted_signal = f.create_dataset(
            "predicted_signal", shape=(len(dataset), *signal_shape), dtype="f"
        )
        target_classifier_embedding = f.create_dataset(
            "target_classifier_embedding",
            shape=(len(dataset), *classifier_embedding_shape),
            dtype="f",
        )
        predicted_classifier_embedding = f.create_dataset(
            "predicted_classifier_embedding",
            shape=(len(dataset), *classifier_embedding_shape),
            dtype="f",
        )
        target_classifier_pred = f.create_dataset(
            "target_classifier_pred", shape=(len(dataset), *classifier_pred_shape), dtype="f"
        )
        predicted_classifier_pred = f.create_dataset(
            "predicted_classifier_pred", shape=(len(dataset), *classifier_pred_shape), dtype="f"
        )

        print(f"Generating waveforms using {device}...")

        for i, batch in enumerate(tqdm(loader, file=sys.stdout)):
            start = i * batch_size
            end = start + len(batch["signal"])

            # target
            target_waveform[start:end] = batch["waveform"].numpy()
            target_signal[start:end] = batch["signal"].numpy()

            # pred
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_signal = edm.evaluate(batch)
            predicted_signal[start:end] = pred_signal.cpu().numpy()
            pred_waveform = config.representation.invert_representation(pred_signal)
            predicted_waveform[start:end] = pred_waveform

            # classifier
            if type(config.representation) == type(classifier_config.representation):
                target_classifier_input = batch["signal"]
                predicted_classifier_input = pred_signal
            else:
                target_classifier_input = th.tensor(
                    classifier_config.representation.get_representation(batch["waveform"]),
                    device=device,
                )
                predicted_classifier_input = th.tensor(
                    classifier_config.representation.get_representation(pred_waveform),
                    device=device,
                )

            target_embedding = classifier.embed(target_classifier_input)
            target_classifier_embedding[start:end] = target_embedding.cpu().numpy()
            target_classifier_pred[start:end] = (
                classifier.output_layer(target_embedding).cpu().numpy()
            )
            pred_embedding = classifier.embed(predicted_classifier_input)
            predicted_classifier_embedding[start:end] = pred_embedding.cpu().numpy()
            predicted_classifier_pred[start:end] = (
                classifier.output_layer(pred_embedding).cpu().numpy()
            )

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train or test)")
    parser.add_argument(
        "--outputdir",
        type=str,
        default="evaluation",
        help="Output subdirectory in the output directory",
    )
    parser.add_argument(
        "--config", type=str, default="LatentSpectrogramConfig", help="Config class for the EDM"
    )
    parser.add_argument(
        "--classifier_config",
        type=str,
        default="SpectrogramClassificationConfig",
        help="Config class for the classifier",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--edm_checkpoint",
        type=str,
        default="Latent-EDM-LogSpectrogram/best.ckpt",
        help="EDM checkpoint",
    )
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default="Classifier-LogSpectrogram/best.ckpt",
        help="Classifier checkpoint",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        default="Autoencoder-32x96x4-LogSpectrogram/best.ckpt",
        help="Optional autoencoder checkpoint. Needed for Latent-EDM.",
    )
    args = parser.parse_args()
    if "latent" not in args.config.lower():
        args.autoencoder_checkpoint = None

    config = getattr(conf, args.config)()
    classifier_config = getattr(conf, args.classifier_config)()
    predict(
        args.split,
        args.outputdir,
        config,
        classifier_config,
        args.batch_size,
        args.edm_checkpoint,
        args.classifier_checkpoint,
        args.autoencoder_checkpoint,
    )
