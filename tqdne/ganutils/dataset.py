import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L


class WFDataModule(L.LightningDataModule):
    def __init__(self, wfs_file, attr_file, v_names, batch_size, train_ratio):
        super().__init__()
        self.v_names = v_names
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.wfs = np.load(wfs_file)
        n = (self.wfs.shape[0] // batch_size) * batch_size
        m = int((n * self.train_ratio) / batch_size) * batch_size
        wfs_train = self.wfs[:m]
        wfs_val = self.wfs[m: n]
        self.df_attr = pd.read_csv(attr_file)[v_names]

        self.data_train = WaveformDataset(wfs_train, self.df_attr, self.v_names)
        self.data_val = WaveformDataset(wfs_val, self.df_attr, self.v_names)

    def get_wfs(self) -> np.ndarray:
        return np.copy(self.wfs)

    def get_attr(self) -> pd.DataFrame:
        return self.df_attr.copy()

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=23)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=23)


class WaveformDataset(Dataset):
    def __init__(self, wfs, df_attr, v_names):
        self.ws = wfs.copy()
        print("normalizing data ...")
        wfs_norm = np.max(np.abs(wfs), axis=1)  # 2)
        self.cnorms = wfs_norm.copy()
        wfs_norm = wfs_norm[:, np.newaxis]
        self.wfs = wfs / wfs_norm
        lc_m = np.log10(self.cnorms)
        max_lc_m = np.max(lc_m)
        min_lc_m = np.min(lc_m)
        self.ln_cns = 2.0 * (lc_m - min_lc_m) / (max_lc_m - min_lc_m) - 1.0

        self.vc_lst = []
        for v_name in v_names:
            v = df_attr[v_name].to_numpy()
            v = (v - v.min()) / (v.max() - v.min())
            # reshape conditional variables
            vc = np.reshape(v, (v.shape[0], 1))
            print("vc shape", vc.shape)
            # 3. store conditional variable
            self.vc_lst.append(vc)

    def __len__(self):
        return self.ws.shape[0]

    def __getitem__(self, index):
        vc_b = [v[index, :] for v in self.vc_lst]
        return (self.wfs[index], self.ln_cns[index], vc_b)
