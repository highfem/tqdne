import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L


class WFDataModule(L.LightningDataModule):
    def __init__(self, wfs_file, attr_file, wfs_expected_size, v_names, batch_size, train_ratio):
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
        self.v_names = v_names
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.wfs = np.load(wfs_file)
        self.df_attr = pd.read_csv(attr_file)[v_names]

        # Reformatting waveforms
        if self.wfs.shape[1] != wfs_expected_size:
            n = wfs_expected_size - self.wfs.shape[1]
            repeat_column = self.wfs[:, 0].reshape(-1, 1)
            filling = repeat_column * np.ones((self.wfs.shape[0], n))
            self.wfs = np.concatenate((filling, self.wfs), axis=1)
        assert self.wfs.shape[1] == wfs_expected_size, "Reshaping failed"

        # Preventing bad division by batch_size
        n = (self.wfs.shape[0] // batch_size) * batch_size
        m = int((n * self.train_ratio) / batch_size) * batch_size
        wfs_train = self.wfs[:m]
        wfs_val = self.wfs[m: n]
        
        self.data_train = WaveformDataset(wfs_train, self.df_attr, self.v_names)
        self.data_val = WaveformDataset(wfs_val, self.df_attr, self.v_names)

    def get_wfs(self) -> np.ndarray:
        """
        Get a copy of the waveforms.

        Returns:
            np.ndarray: A copy of the waveforms.

        """
        return np.copy(self.wfs)

    def get_attr(self) -> pd.DataFrame:
        """
        Get a copy of the attribute data.

        Returns:
            pd.DataFrame: A copy of the attribute data.

        """
        return self.df_attr.copy()
    
    def getSignalFromDecomp(self, wfs: np.array, lcn: np.array):
        """
        Get the signal from decomposition.

        Args:
            wfs (np.array): Array of waveforms.
            lcn (np.array): Array of log compression values.

        Returns:
            np.array: The signal obtained from decomposition.

        """
        return self.data_train.getSignalFromDecomp(wfs, lcn)

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


class WaveformDataset(Dataset):
    def __init__(self, wfs, df_attr, v_names):
        """
        Initialize the WaveformDataset.

        Args:
            wfs (np.array): Array of waveforms.
            df_attr (pd.DataFrame): DataFrame containing attribute data.
            v_names (list): List of attribute names.

        """
        print("normalizing data ...")
        self.ws = wfs.copy()
        
        # Normalize wfs
        wfs_norm = np.max(np.abs(wfs), axis=1)
        # self.cnorms = wfs_norm.copy()
        wfs_norm = wfs_norm[:, np.newaxis]
        self.wfs = wfs / wfs_norm

        # Transform normalization
        lc_m = np.log10(wfs_norm)
        self.max_lc_m = np.max(lc_m)
        self.min_lc_m = np.min(lc_m)
        self.ln_cns = 2.0 * (lc_m - self.min_lc_m) / (self.max_lc_m - self.min_lc_m) - 1.0

        self.vc_lst = []
        for v_name in v_names:
            v = df_attr[v_name].to_numpy()
            v = (v - v.min()) / (v.max() - v.min())
            vc = np.reshape(v, (v.shape[0], 1))
            self.vc_lst.append(vc)
        self.vc_lst = np.array(self.vc_lst)

    def getSignalFromDecomp(self, wfs: np.array, lcn: np.array):
        """
        Get the signal from decomposition.

        Args:
            wfs (np.array): Array of waveforms. Shape (N, waveform_size).
            lcn (np.array): Array of log compression values. Shape (N, 1).

        Returns:
            np.array: The signal obtained from decomposition.

        """
        lcn_ = (self.max_lc_m - self.max_lc_m) * (lcn + 1.0) / 2.0 + self.min_lc_m
        lcn_ = 10 ** lcn_
        return lcn_ * wfs

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return self.ws.shape[0]

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index: Index(es) of the item.

        Returns:
            tuple: A tuple containing the waveform, log compression values, and conditional variables.

        """
        vc_b = self.vc_lst[:, index, :]
        return (self.wfs[index], self.ln_cns[index], vc_b)
