import itertools
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch
import tqdm
from scipy import signal
from typing import Type

from tqdne.conf import Config

@staticmethod
def downsample_waveform(waveform, downsample_factor): 
    original_signal_length = waveform.shape[-1]
    waveform_ds = scipy.signal.decimate(waveform, downsample_factor, axis=-1, zero_phase=False) 
    if waveform_ds.shape[-1] > original_signal_length // downsample_factor:
            waveform_ds = waveform_ds[..., :-1]
    return waveform_ds

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


def compute_mean_std_features(datapath, features_keys, indices=None):
    """Compute mean and std of features in a dataset."""
    with h5py.File(datapath, "r") as f:
        stds = []
        means = []
        for key in features_keys:
            if indices is not None:
                mean, std = compute_mean_std(f[key][0, indices])
            else:    
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


def build_dataset(config=Config(), batch_size=1000):
    """Build the dataset."""

    # extract the config information
    output_path = config.datasetdir
    datapath = config.datapath
    features_keys = config.features_keys

    with h5py.File(datapath, "r") as f:
        # remove samples with vs30 <= 0
        mask = f["vs30"][0, :] > 0
        indices = np.arange(len(mask))[mask]

        time = f["time_vector"]
        t = len(time)
        nf = len(features_keys)
        n = len(indices)
        n_train = round(n * config.train_ratio)
        # reset the random state
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices = np.sort(indices[:n_train])
        test_indices = np.sort(indices[n_train:])

        def create_dataset(name, indices):
            processed_path = output_path / Path(name)
            processed_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
            with h5py.File(processed_path, "w") as fout:
                fout.create_dataset("waveforms", (len(indices), config.num_channels, t))
                fout.create_dataset("features", (len(indices), nf))
                for i in tqdm.tqdm(range(0, len(indices), batch_size)):
                    batch_waveforms = f["waveforms"][:, :, indices[i : i + batch_size]].transpose(2, 0, 1)
                    batch_waveforms = np.nan_to_num(batch_waveforms)
                    fout["waveforms"][i : i + batch_size] = batch_waveforms
                    batch_features = np.array([f[key][0, indices[i : i + batch_size]] for key in features_keys]).T
                    fout["features"][i : i + batch_size] = batch_features

        create_dataset(config.data_train, train_indices) 
        create_dataset(config.data_test, test_indices) 
        

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, representation, n = 1024 * 8, t = 5472):
        super().__init__()
        self.n = n
        self.t = t
        self.representation = representation
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

class LowFreqDataset(torch.utils.data.Dataset):
    def __init__(self, n = 1024 * 8, t = 5472, fs=100):
        super().__init__()
        self.n = n
        self.t = t
        self.fs = fs

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        noise = np.random.randn(self.t)
        time = np.arange(self.t) / self.fs
        # random scalar parameters between -3 and 3
        a = np.random.rand() * 6 - 3
        b = np.random.rand() * 6 - 3
        c = np.random.rand() * 6 - 3
        d = np.random.rand() * 6 - 3

        sin1 = a * np.sin(2 * np.pi * 0.5*(3+d) * time)
        sin2 = b * np.sin(2 * np.pi * 2*(3+c) * time)
        sin3 = c * np.sin(2 * np.pi * 4*(3+d+3+b) * time)
        sin4 = d * np.sin(2 * np.pi * 7*(3+a) * time)

        signal = sin1+sin2/2+sin4/4 + sin2*np.exp(-time) + sin3/(1+time) + sin4*time/5 + noise
        cond = np.array([a, b, c, d])

        return {
            "repr": torch.tensor(signal.reshape(1, -1), dtype=torch.float32),
            "cond": torch.tensor(cond, dtype=torch.float32),
        }
    
    
class RepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, data_repr, max_amplitude=None, pad=None, downsample=1):
        super().__init__()
        self.h5_path = h5_path
        self.representation = data_repr

        self.file = h5py.File(h5_path, "r", locking=False)
        self.features = self.file["features"][:]
        try:
            self.waveforms = self.file["waveforms"]
        except KeyError:
            self.waveforms = self.file["waveform"] 
        
        try:
            self.features_means = self.file["feature_means"][:]
            self.features_stds = self.file["feature_stds"][:] 
        except KeyError:
            pass
        
        # Remove the third feature (log10snr)
        if self.features.shape[1] == 5:
            self.features = self.features[:, [0, 1, 3, 4]] 

        # Remove samples with VS30 <= 0
        idxs = self.features[:, 3] > 0
        self.features = self.features[idxs]
        self.waveforms = self.waveforms[idxs]

        # TODO: REMOVE - ONLY TO TEST MODEL WITH NO ENVELOPE
        # Take only samples with amplitude < max_amplitude
        if max_amplitude is not None:
            idxs_low_amp = np.abs(self.waveforms).max(axis=(1, 2)) < max_amplitude
            self.features = self.features[idxs_low_amp]
            self.waveforms = self.waveforms[idxs_low_amp]

        self.n = len(self.features)
        assert self.n == len(self.waveforms)
        self.pad = pad
        self.downsample = downsample


    def __del__(self):
        pass
        #if not self.in_memory:
        #    self.file.close()

    def __len__(self):
        return self.n

    def __getitem__(self, index):

        signal = self.waveforms[index]
        features = self.features[index]


        if self.pad:
            if self.pad <= signal.shape[-1]:
                signal = signal[:, : self.pad]
            else:
                # Pad at the end with the last value
                signal = np.concatenate([signal, signal[:, -1, np.newaxis] * np.ones((signal.shape[0], self.pad - signal.shape[-1]))], axis=-1)                

        if self.downsample > 1:
            signal = downsample_waveform(signal, self.downsample)

        repr = self.representation.get_representation(signal)    

        return {
            "waveform": torch.tensor(signal, dtype=torch.float32),
            "repr": torch.tensor(repr, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }
    
    # TODO: maybe should be moved to SampleDataset
    def get_waveforms_by_cond_input(self, cond_input):
        """
        Get waveforms based on conditional input. If conditional input is None, all waveforms are returned.

        Args:
            cond_input (ndarray, optional): Conditional input array.

        Returns:
            ndarray: Array of waveforms based on the conditional input.
        """


        if cond_input is None:
            idxs = np.ones(self.features.shape[0], dtype=bool)
        else:
            idxs = np.where(np.all(self.features[:, None] == cond_input, axis=2))[0]

        return self[idxs]        
    
    def get_data_by_bin(self, magnitude_bin: tuple, distance_bin: tuple, is_shallow_crustal: bool = None):
        bins_idxs = (self.features[:, 0] >= distance_bin[0]) & (self.features[:, 0] < distance_bin[1]) & (self.features[:, 2] >= magnitude_bin[0]) & (self.features[:, 2] < magnitude_bin[1])
        if is_shallow_crustal is not None:
            bins_idxs = bins_idxs & (self.features[:, 1] == is_shallow_crustal)
        if np.any(bins_idxs):
            return self[bins_idxs]
        raise ValueError("No data in the given bins")
        

class ClassifierDataset(RepresentationDataset):
    def __init__(self, h5_path, data_repr, mag_bins, dist_bins, max_amplitude=None, pad=None, downsample=1):
        """
        Initialize the ClassifierDataset.

        Args:
            h5_path (str): The path to the HDF5 file.
            data_repr (str): The representation of the data.
            mag_bins (list): The magnitude bins.
            dist_bins (list): The distance bins.
            max_amplitude (float, optional): The maximum amplitude. Defaults to None.
            pad (int, optional): The padding size. Defaults to None.
            downsample (int, optional): The downsampling factor. Defaults to 1.
        """
        super().__init__(h5_path, data_repr, max_amplitude, pad, downsample)
        self.mag_bins = mag_bins
        self.dist_bins = dist_bins
        self.bin_mapping = {f"{i}_{j}": idx for idx, (i, j) in enumerate(np.ndindex((len(dist_bins), len(mag_bins))))}
        
    def __del__(self):
        pass
        #if not self.in_memory:
        #    self.file.close()

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.n

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            dict: The item from the dataset. In addition to the waveform and representation, the item also contains the class label.
        """
        item = super().__getitem__(index)
        item.update({
            "classes": torch.tensor(self._get_class_label(self.features[index]), dtype=torch.long)
        })

        return item
    
    def _get_class_label(self, features):
        """
        Get the class label for a given set of features.

        Args:
            features (list): The features.

        Returns:
            int: The class label.
        """
        mag, dist = features[2], features[0]
        for i, dist_bin in enumerate(self.dist_bins):
            for j, mag_bin in enumerate(self.mag_bins):
                if dist >= dist_bin[0] and dist < dist_bin[1] and mag >= mag_bin[0] and mag < mag_bin[1]:
                    return self.bin_mapping[f"{i}_{j}"]
    
    def get_num_classes(self):
        """
        Get the number of classes in the dataset.

        Returns:
            int: The number of classes.
        """
        return len(self.bin_mapping) if self.bin_mapping else 0
    
    def get_class_weights(self):
        """
        Calculate the class weights based on the frequency of each class label in the dataset.

        Returns:
            torch.Tensor: The class weights.
        """
        class_counts = torch.bincount(torch.tensor([self._get_class_label(features) for features in self.features]))
        class_weights = 1. / class_counts.float()
        return class_weights / class_weights.sum()