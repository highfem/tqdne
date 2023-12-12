import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L

def bandpass_filter(n, p):
    f = np.fft.fftfreq(n)
    # f = np.fft.fftshift(f)
    filt = np.exp(- (np.abs(f) - p)**2 / 0.0001)
    return filt

def random_stationary_signal(n = 1024, sigma = 1):
    noise = sigma * np.random.randn(n)
    p = np.random.rand(1) / 16
    # apply a bandpass filter
    noise_hat = np.fft.fft(noise)
    filt = bandpass_filter(n, p)
    sig_hat = noise_hat * filt
    sig = np.fft.ifft(sig_hat)
    sig = sig.real
    return p*16, sig

class StationarySignalDM(L.LightningDataModule):
    def __init__(self, data_size, wfs_len, batch_size, train_ratio):
        """
        Initialize the WFDataModule.

        Args:
            wfs_file (str): File path to the waveforms file.
            attr_file (str): File path to the attribute file.
            wfs_expected_size (int): Expected size of the waveforms.
            v_names (list): List of attribute names.
            batch_size (int): Batch size for training and validation.
            train_ratio (float): Ratio of training data to total data.

        """
        super().__init__()
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        # Generate random stationary signals
        self.p = []
        self.wfs = []
        for _ in range(data_size):
            p, sig = random_stationary_signal(wfs_len)
            self.p.append(p)
            self.wfs.append(sig)
        self.p = np.array(self.p)
        print("shape p:", self.p.shape)
        self.wfs = np.array(self.wfs)

        # Preventing bad division by batch_size
        n = (self.wfs.shape[0] // batch_size) * batch_size
        m = int((n * self.train_ratio) / batch_size) * batch_size
        wfs_train = self.wfs[:m]
        p_train = self.p[:m]
        wfs_val = self.wfs[m: n]
        p_val = self.p[m: n]
        
        self.data_train = StationarySignalDataset(wfs_train, p_train)
        self.data_val = StationarySignalDataset(wfs_val, p_val)

    def get_wfs(self):
        return self.wfs.copy()
    
    def get_conds(self):
        return self.p.copy()

    def train_dataloader(self):
        """
        Get the training dataloader.

        Returns:
            DataLoader: The training dataloader.

        """
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=23)

    def val_dataloader(self):
        """
        Get the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.

        """
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=23)


class StationarySignalDataset(Dataset):
    def __init__(self, wfs, p):
        """
        Initialize the WaveformDataset.

        Args:
            wfs (np.array): Array of waveforms.
            df_attr (pd.DataFrame): DataFrame containing attribute data.
            v_names (list): List of attribute names.

        """
        print("normalizing data ...")
        self.wfs = wfs
        self.p = p

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return self.wfs.shape[0]

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index: Index(es) of the item.

        Returns:
            tuple: A tuple containing the waveform, log compression values, and conditional variables.

        """
        w = self.wfs[index]
        return (w, np.full(w.shape,np.nan), self.p[index])
