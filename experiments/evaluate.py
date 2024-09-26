"""Evaluate the trained diffusion models.

This script generates waveforms using the same conditional features as the dataset. 
The generated waveforms are saved along with original waveforms, conditional features, and classifier outputs in an HDF5 file. 
The created file can be read by the corresponding notebook to compute metrics and create plots.
"""

from pathlib import Path
import sys
from argparse import ArgumentParser

import torch
import torch as th
from h5py import File
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from tqdne.conf import Config
from tqdne.dataset import RepresentationDataset
from tqdne import utils
from tqdne.representations import to_torch, to_numpy


@torch.no_grad()
def predict(
    outputdir,
    train_datapath,
    test_datapath,
    batch_size,
    diffusion_model_checkpoint,
    classifier_checkpoint,
    debug=False,
    config=Config()
):
    """Evaluate the trained EDM model.

    Parameters
    ----------
    outputdir : str
        Output directory for the evaluation results.
    train_datapath : str
        Path to the training dataset.
    test_datapath : str 
        Path to the test dataset.
    batch_size : int
        Batch size used for the generation.
    diffusion_model_checkpoint : str
        Saved diffusion model checkpoint. Relative to the output directory.
    classifier_checkpoint : str
        Saved classifier model checkpoint. Relative to the output directory.
    debug : bool, optional
        If True, only evaluate the first batch, by default False
    config : Config, optional
        Configuration object, by default Config()
    """ 

    if "downsampling" in diffusion_model_checkpoint:
        downsampling = int(diffusion_model_checkpoint.split("downsampling:")[1].split("_")[0])
    else:
        downsampling = 1

    signal_length = config.signal_length // downsampling
    fs = config.fs // downsampling
    config.signal_length = signal_length
    config.fs = fs    

    
    # Load models
    print("Loading model...")
    device = "cuda" if th.cuda.is_available() else "cpu"
    diffusion_model, diffusion_data_repr, _ = utils.load_model(Path(diffusion_model_checkpoint), print_info=False, device=device, signal_length=config.signal_length)
    classifier, classifier_data_repr, _ = utils.load_model(Path(classifier_checkpoint), print_info=False, device=device, signal_length=config.signal_length)

    # Load datasets
    if debug:
        batch_size = 16
    test_dataset = RepresentationDataset(Path(test_datapath), diffusion_data_repr, pad=config.signal_length*downsampling, downsample=downsampling)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_dataset = RepresentationDataset(Path(train_datapath), diffusion_data_repr, pad=config.signal_length*downsampling, downsample=downsampling)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    full_dataset = ConcatDataset([train_dataset, test_dataset])
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
 

    # get shapes 
    batch = next(iter(full_dataloader))
    repr_shape = diffusion_data_repr.get_representation(batch["waveform"]).shape[1:]
    waveform_shape = batch["waveform"].shape[1:]
    classifier_embedding = classifier.get_embeddings(batch["waveform"], classifier_data_repr)
    classifier_embedding_shape = classifier_embedding.shape[1:]
    classifier_pred_shape = classifier(to_torch(classifier_data_repr.get_representation(batch["waveform"]), device=device)).shape[1:]

    outputdir = config.outputdir / outputdir
    outputdir.mkdir(exist_ok=True)
    for split in ["train", "test"]:
        print(f"Evaluting {split} split...")
        if split == "train":
            dataset = train_dataset
            loader = train_dataloader
        else:
            dataset = test_dataset
            loader = test_dataloader

        with File(outputdir / f"{split}.h5", "w") as f:
            for i, key in enumerate(config.features_keys):
                f.create_dataset(key, data=dataset.features[i])

            target_waveform = f.create_dataset(
                "target_waveform", shape=(len(dataset), *waveform_shape), dtype="f"
            )
            predicted_waveform = f.create_dataset(
                "predicted_waveform", shape=(len(dataset), *waveform_shape), dtype="f"
            )
            target_representation = f.create_dataset(
                "target_representation", shape=(len(dataset), *repr_shape), dtype="f"
            )
            predicted_representation = f.create_dataset(
                "predicted_representation", shape=(len(dataset), *repr_shape), dtype="f"
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
                if debug and i > 0:
                    break

                start = i * batch_size
                end = start + len(batch["repr"])

                # target
                target_waveform[start:end] = batch["waveform"].numpy()
                target_representation[start:end] = batch["repr"].numpy()

                # pred
                batch = {k: v.to(device) for k, v in batch.items()}
                pred_signal = diffusion_model.evaluate(batch)
                predicted_representation[start:end] = pred_signal.cpu().numpy()
                pred_waveform = diffusion_data_repr.invert_representation(pred_signal)
                predicted_waveform[start:end] = pred_waveform

                # classifier
                target_classifier_input = to_torch(
                    classifier_data_repr.get_representation(batch["waveform"]),
                    device=device,
                )
                predicted_classifier_input = to_torch(
                    classifier_data_repr.get_representation(pred_waveform),
                    device=device,
                )

                target_embedding = classifier.get_embeddings(target_classifier_input)
                target_classifier_embedding[start:end] = to_numpy(target_embedding)
                target_classifier_pred[start:end] = (
                    classifier.get_predictions(target_embedding, from_embeddings=True)
                )
                pred_embedding = classifier.get_embeddings(predicted_classifier_input)
                predicted_classifier_embedding[start:end] = to_numpy(pred_embedding)
                predicted_classifier_pred[start:end] = (
                    classifier.get_predictions(pred_embedding, from_embeddings=True)
                )

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--outputdir",
        type=str,
        help="Output subdirectory in the output directory",
    )
    parser.add_argument("--train_datapath", type=str, default="datasets/data_train.h5", help="Train dataset path")
    parser.add_argument("--test_datapath", type=str, default="datasets/data_test.h5", help="Test dataset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default="outputs/classifier-2D-32Chan-(1, 2, 4, 8)Mult-2ResBlocks-4AttHeads_LogSpectrogram-stft_ch:128-hop_size:32/name=0_epoch=27-val_loss=0.92.ckpt",
        help="Classifier checkpoint",
    )
    parser.add_argument(
        "--diffusion_model_checkpoint",
        type=str,
        default="outputs/ddim-pred:sample-2D-downsampling:2_LogSpectrogram-stft_ch:128-hop_size:32/name=0_epoch=126-val_loss=0.01.ckpt",
        help="Diffusion model checkpoint",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
    )
    args = parser.parse_args()


    predict(
        args.outputdir,
        args.train_datapath,
        args.test_datapath,
        args.batch_size,
        args.diffusion_model_checkpoint,
        args.classifier_checkpoint,
        args.debug
    )