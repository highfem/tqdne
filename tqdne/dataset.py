import numpy as np
import torch as th
from h5py import File
from seisbench.data import WaveformDataset


class Dataset(th.utils.data.Dataset):
    """Dataset for seismic data stored in an HDF5 file.

    Parameters:
    -----------
    h5_path : Path
        Path to the dataset HDF5 file.
    representaion : Representation
        Representation object to transform the waveforms.
    cut : int, optional
        Cut the waveforms to this length.
    cond : bool, optional
        If True, the dataset will return the normalized features as condition.
    split : str, optional
        The split of the dataset. One of "train", "test", or "full".
    """

    def __init__(self, datapath, representaion, cut=None, cond=False, split="train"):
        super().__init__()
        self.representation = representaion
        self.cut = cut
        self.cond = cond

        self.file = File(datapath, "r")
        self.waveforms = self.file["waveforms"]
        self.cond = self.file["normalized_features"] if cond else None

        # train test split
        indices = np.arange(len(self.waveforms))
        rng = np.random.default_rng(seed=42)
        shuffled_indices = rng.permutation(indices)
        num_train_samples = int(len(indices) * 0.85)
        num_val_samples = int(len(indices) * 0.9)
        if split == "full":
            self.indices = indices
        elif split == "train":
            self.indices = shuffled_indices[:num_train_samples]
        elif split == "validation":
            self.indices = shuffled_indices[num_train_samples:num_val_samples]
        elif split == "test":
            self.indices = shuffled_indices[num_val_samples:]
        else:
            raise ValueError(f"Unknown split {split}")

    def sorted_indices(self):
        return np.sort(self.indices)

    def get_feature(self, key):
        return self.file[key][:][self.indices]

    def __del__(self):
        self.file.close()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        waveform = self.file["waveforms"][self.indices[index]]

        if self.cut:
            waveform = waveform[:, : self.cut]

        signal = self.representation.get_representation(waveform)

        out = {
            "waveform": th.tensor(waveform, dtype=th.float32),
            "signal": th.tensor(signal, dtype=th.float32),
        }

        if self.cond:
            out["cond"] = th.tensor(self.cond[self.indices[index]], dtype=th.float32)

        return out


class ClassificationDataset(Dataset):
    def __init__(self, h5_path, representaion, mag_bins, dist_bins, cut=None, split="train"):
        super().__init__(h5_path, representaion, cut=cut, cond=False, split=split)

        # compute labels
        # labels = dist_bin * len(mag_bins) + mag_bin
        dist = self.file["hypocentral_distance"]
        mag = self.file["magnitude"]
        self.labels = (
            (np.digitize(dist, dist_bins) - 1) * (len(mag_bins) - 1)
            + np.digitize(mag, mag_bins)
            - 1
        )        
        self._split = split
        self._num_classes = (len(mag_bins) - 1) * (len(dist_bins) - 1)        

    def get_class_weights(self):
        assert self._num_classes == len(np.unique(self.labels))        
        return th.tensor(
            [1 / (self.labels == l).sum() for l in range(self._num_classes)], dtype=th.float32
        )

    def __getitem__(self, index):
        out = super().__getitem__(index)
        out["label"] = th.tensor(self.labels[self.indices[index]], dtype=th.long)
        return out


class SeisbenchDataset(th.utils.data.Dataset):
    def __init__(self, obs_path, syn_path, representaion, cut, cond=False, training=True):
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
        indices = np.nonzero(mask)[0]

        # train test split
        rng = np.random.default_rng(seed=42)
        shuffled_indices = rng.permutation(indices)
        num_train_samples = int(len(indices) * 0.9)
        if training:
            self.indices = shuffled_indices[:num_train_samples]
        else:
            self.indices = shuffled_indices[num_train_samples:]

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
            "waveform": th.tensor(obs, dtype=th.float32),
            "cond_waveform": th.tensor(syn, dtype=th.float32),
            "signal": th.tensor(signal, dtype=th.float32),
            "cond_signal": th.tensor(cond_signal, dtype=th.float32),
        }
