from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from scipy import signal
from seisbench.data import WaveformDataset

from tqdne.conf import Config


def compute_mean_std(array):
    """Compute mean and std of an array.

    This function remove all nan, inf and -inf values before computing mean and std.

    Parameters
    ----------
    array : np.array

    Returns
    -------
    mean : float
        Mean of the array.
    std : float
        Standard deviation of the array.
    """
    array = array[np.isfinite(array)]
    return np.mean(array), np.std(array)


def compute_mean_std_features(datapath, features_keys):
    """Compute mean and std of features in a dataset."""
    with h5py.File(datapath, "r") as f:
        stds = []
        means = []
        for key in features_keys:
            mean, std = compute_mean_std(f[key][0])
            means.append(mean)
            stds.append(std)

    return np.array(means), np.array(stds)


def extract_sample_from_h5file(f, idx, config=Config()):
    """Extract a sample from a h5 file.

    Args:
        f: h5 file
        idx: index of the sample to extract

    """
    # time = f["time_vector"][:]
    waveform = f["waveforms"][:, :, idx]
    # replace nan with 0
    waveform = np.nan_to_num(waveform)
    features = [np.nan_to_num(f[key][0, idx]) for key in config.features_keys]
    return waveform, np.array(features)


def build_dataset(config=Config()):
    """Build the dataset."""

    # extract the config information
    output_path = config.datasetdir
    datapath = config.datapath
    features_keys = config.features_keys

    # Create the filter
    sos = signal.butter(**config.params_filter, fs=config.fs, output="sos")

    with h5py.File(datapath, "r") as f:
        time = f["time_vector"][:]
        t = len(time)
        nf = len(features_keys)
        n = f["waveforms"].shape[2]
        n_train = 1024 * (128 + 64)
        # reset the random state
        np.random.seed(42)
        permutation = np.random.permutation(n)
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]
        means, stds = compute_mean_std_features(datapath, features_keys)

        def create_dataset(name, indices):
            processed_path = output_path / Path(name)
            with h5py.File(processed_path, "w") as fout:
                fout.create_dataset("time", data=time)
                fout.create_dataset("feature_means", data=means)
                fout.create_dataset("feature_stds", data=stds)
                waveforms = fout.create_dataset("waveform", (len(indices), 3, t))
                filtered = fout.create_dataset("filtered", (len(indices), 3, t))
                featuress = fout.create_dataset("features", (len(indices), nf))
                for i, idx in tqdm.tqdm(enumerate(indices), total=len(indices)):
                    waveform, features = extract_sample_from_h5file(f, idx)
                    filtered[i] = np.array([signal.sosfilt(sos, channel) for channel in waveform])
                    waveforms[i] = waveform
                    featuress[i] = features

        create_dataset(config.data_upsample_train, train_indices)
        create_dataset(config.data_upsample_test, test_indices)


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n=1024 * 8, t=5472):
        super().__init__()
        self.n = n
        self.t = t
        self.lp = signal.butter(10, 1, "hp", fs=100, output="sos")
        self.bp = signal.butter(2, [0.25, 10], "bp", fs=100, output="sos")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        noise = np.random.randn(self.t)
        x = signal.sosfilt(self.bp, noise)
        lowpass = signal.sosfilt(self.lp, x)  # + 0.1 * x

        return {
            "signal": torch.tensor(x.reshape(1, -1), dtype=torch.float32),
            "cond_signal": torch.tensor(lowpass.reshape(1, -1), dtype=torch.float32),
        }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, representaion, cut=None, cond=False):
        super().__init__()
        self.h5_path = h5_path
        self.cut = cut
        self.cond = cond
        self.representation = representaion

        self.file = h5py.File(h5_path, "r")
        self.waveform = self.file["waveform"]
        if cond:
            self.features = self.file["features"][:]
            self.features_means = self.file["feature_means"][:]
            self.features_stds = self.file["feature_stds"][:]

            # hack: set inf log10snr to largest non-inf value
            log10snr = self.features[:, 2]
            log10snr[log10snr == np.inf] = np.max(log10snr[log10snr != np.inf])
            self.features[:, 2] = log10snr

    def __del__(self):
        self.file.close()

    def __len__(self):
        return len(self.waveform)

    def __getitem__(self, index):
        waveform = self.waveform[index]

        if self.cut:
            waveform = waveform[:, : self.cut]

        signal = self.representation.get_representation(waveform)

        if not self.cond:
            return {
                "waveform": torch.tensor(waveform, dtype=torch.float32),
                "signal": torch.tensor(signal, dtype=torch.float32),
            }

        features = self.features[index]
        features = (features - self.features_means) / self.features_stds

        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }


class UpsamplingDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, cut=None, cond=False, config=Config()):
        super().__init__()
        self.h5_path = h5_path
        self.cut = cut
        self.cond = cond
        self.sigma_in = config.sigma_in

        self.file = h5py.File(h5_path, "r")
        self.waveform = self.file["waveform"]
        self.filtered = self.file["filtered"]
        if cond:
            self.features = self.file["features"][:]
            self.features_means = self.file["feature_means"][:]
            self.features_stds = self.file["feature_stds"][:]

            # hack: set inf log10snr to largest non-inf value
            log10snr = self.features[:, 2]
            log10snr[log10snr == np.inf] = np.max(log10snr[log10snr != np.inf])
            self.features[:, 2] = log10snr

    def __del__(self):
        self.file.close()

    def __len__(self):
        return len(self.waveform)

    def __getitem__(self, index):
        waveform = self.waveform[index]
        filtered = self.filtered[index]

        # normalize
        scale = np.abs(filtered).max()
        waveform = waveform / scale / 5
        filtered = filtered / scale * 2

        # add noise to filtered
        filtered += np.random.randn(*filtered.shape) * self.sigma_in

        # features
        features = self.features[index]
        features = (features - self.features_means) / self.features_stds

        if self.cut:
            signal = waveform[:, : self.cut]
            cond_signal = filtered[:, : self.cut]
        else:
            signal = waveform
            cond_signal = filtered

        return {
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cond_signal": torch.tensor(cond_signal, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }


class SeisbenchDataset(torch.utils.data.Dataset):
    def __init__(self, obs_path, syn_path, representaion, cut, cond=False, config=Config()):
        super().__init__()
        self.cond = cond
        self.cut = cut
        self.representation = representaion
        self.obs_data = WaveformDataset(obs_path)
        self.syn_data = WaveformDataset(syn_path)

        # filter out bad samples
        def save_filter(fn):
            def filter(x):
                try:
                    return all(fn(np.array(eval(x))))
                except Exception:
                    return True

            return filter

        snr_mask = self.obs_data.metadata["trace_snr"].apply(save_filter(lambda x: x > 1.5))
        snr_mask &= self.syn_data.metadata["trace_snr"].apply(save_filter(lambda x: x > 1.5))
        ratio_mask = self.obs_data.metadata["data_ratio"].apply(save_filter(lambda x: x < 10))
        ratio_mask &= self.syn_data.metadata["data_ratio"].apply(save_filter(lambda x: x < 10))
        mask = snr_mask & ratio_mask
        self.indices = np.nonzero(mask)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        obs = self.obs_data.get_sample(self.indices[index])[0]
        syn = self.syn_data.get_sample(self.indices[index])[0]

        if self.cut:
            obs = obs[:, : self.cut]
            syn = syn[:, : self.cut]

            # zero pad if necessary
            if obs.shape[1] < self.cut:
                obs = np.pad(obs, ((0, 0), (0, self.cut - obs.shape[1])), "constant")
            if syn.shape[1] < self.cut:
                syn = np.pad(syn, ((0, 0), (0, self.cut - syn.shape[1])), "constant")

        obs = np.nan_to_num(obs)
        syn = np.nan_to_num(syn)

        signal = self.representation.get_representation(obs)
        cond_signal = self.representation.get_representation(syn)

        return {
            "waveform": torch.tensor(obs, dtype=torch.float32),
            "cond_waveform": torch.tensor(syn, dtype=torch.float32),
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cond_signal": torch.tensor(cond_signal, dtype=torch.float32),
        }
