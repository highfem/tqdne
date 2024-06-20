"""Evaluate the latent EDM model.

This script generates waveforms using the same conditional features as the dataset. 
The generated waveforms are saved along with original waveforms, conditional features, and classifier outputs in an HDF5 file. 
The created file can be read by the corresponding notebook to compute metrics and create plots.
"""

import sys
from argparse import ArgumentParser

import torch
import torch as th
from h5py import File
from torch.utils.data import DataLoader
from tqdm import tqdm

import tqdne.config as conf
from tqdne.autoencoder import LithningAutoencoder
from tqdne.classifier import LithningClassifier
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM


@torch.no_grad()
def predict(
    split,
    config,
    batch_size,
    edm_checkpoint,
    classifier_checkpoint,
    autoencoder_checkpoint,
):
    print(f"Predicting {split} set...")

    dataset = Dataset(config.datapath, config.representation, cut=config.t, cond=True, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    print("Loading model...")

    device = "cuda" if th.cuda.is_available() else "cpu"
    if autoencoder_checkpoint is not None:
        autoencoder = LithningAutoencoder.load_from_checkpoint(
            config.outputdir / autoencoder_checkpoint
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
    batch = {k: v.to("cuda") for k, v in batch.items()}
    signal_shape = batch["signal"].shape[1:]
    waveform_shape = batch["waveform"].shape[1:]
    classifier_embedding = classifier.embed(batch["signal"])
    classifier_embedding_shape = classifier_embedding.shape[1:]
    classifier_pred_shape = classifier.output_layer(classifier_embedding).shape[1:]

    outputdir = config.outputdir / "evaluation"
    outputdir.mkdir(exist_ok=True)
    with File(outputdir / f"{split}.h5", "w") as f:
        f.create_dataset("dist", data=dataset.get_feature("hypocentral_distance"))
        f.create_dataset("mag", data=dataset.get_feature("magnitude"))
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
            batch = {k: v.to("cuda") for k, v in batch.items()}
            target_embedding = classifier.embed(batch["signal"])
            target_classifier_embedding[start:end] = target_embedding.cpu().numpy()
            target_classifier_pred[start:end] = (
                classifier.output_layer(target_embedding).cpu().numpy()
            )

            # pred
            pred_signal = edm.evaluate(batch)
            predicted_signal[start:end] = pred_signal.cpu().numpy()
            predicted_waveform[start:end] = config.representation.invert_representation(pred_signal)
            pred_embedding = classifier.embed(pred_signal)
            predicted_classifier_embedding[start:end] = pred_embedding.cpu().numpy()
            predicted_classifier_pred[start:end] = (
                classifier.output_layer(pred_embedding).cpu().numpy()
            )
            print("Done", flush=True)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train or test)")
    parser.add_argument(
        "--config", type=str, default="LatentSpectrogramConfig", help="Config class for the EDM"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--edm_checkpoint",
        type=str,
        default="Latent-EDM-LogSpectrogram/0_239-val_loss=1.18e+00.ckpt",
        help="EDM checkpoint",
    )
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default="Classifier-LogSpectrogram/0_21-val_loss=1.09e+00.ckpt",
        help="Classifier checkpoint",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        default="Autoencoder-32x32x4-LogSpectrogram/0_199-val_loss=1.55e-03.ckpt",
        help="Optional autoencoder checkpoint. Needed for Latent-EDM.",
    )
    args = parser.parse_args()

    config = getattr(conf, args.config)()
    predict(
        args.split,
        config,
        args.batch_size,
        args.edm_checkpoint,
        args.classifier_checkpoint,
        args.autoencoder_checkpoint,
    )
