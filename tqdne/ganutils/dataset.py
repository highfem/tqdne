import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L

class WFDataModule(L.LightningDataModule):
    def __init__(self, wfs_file, attr_file, v_names, batch_size, train_ratio):
        super().__init__()
        self.wfs_file = wfs_file
        self.attr_file = attr_file
        self.v_names = v_names
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        wfs = np.load(self.wfs_file)
        n = int(wfs.shape[0] * self.train_ratio)
        wfs_train = wfs[:n]
        wfs_val = wfs[n:]
        df_attr = pd.read_csv(self.attr_file)

        self.data_train = WaveformDataset(wfs, df_attr, self.v_names)
        self.data_val = WaveformDataset(wfs_val, df_attr, self.v_names)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers = 7)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True)

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

        # store All attributes
        # store pandas dict as attribute
        df_meta = df_attr[v_names]

        self.vc_lst = []
        for v_name in v_names:
            v = df_meta[v_name].to_numpy()
            v = (v - v.min()) / (v.max() - v.min())

            # reshape conditional variables
            vc = np.reshape(v, (v.shape[0], 1))
            print("vc shape", vc.shape)
            # 3. store conditional variable
            self.vc_lst.append(vc)

        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v)
        self.vc_lst = vc_b

    def __len__(self):
        return self.ws.shape[0]

    def __getitem__(self, index):
        vc_b = [v[index, :] for v in self.vc_lst]
        return (self.wfs[index], self.ln_cns[index], vc_b)


