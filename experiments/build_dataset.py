import numpy as np
from h5py import File
from tqdm import tqdm

from tqdne.config import Config

if __name__ == "__main__":
    config = Config()

    with File(config.original_datapath, "r") as f:
        # remove samples with vs30 <= 0
        mask = f["vs30"][0, :] > 0
        indices = np.arange(len(mask))[mask]

        with File(config.datapath, "w") as f_new:
            features = []
            for key in config.features_keys:
                feature = f[key][0][mask]
                f_new.create_dataset(key, data=feature)
                features.append(feature)

            features = np.stack(features, axis=1)
            normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
            f_new.create_dataset("normalized_features", data=normalized_features)

            channels, t, samples = f["waveforms"].shape

            f_new.create_dataset("waveforms", (len(indices), channels, t))
            batch_size = 1000
            for i in tqdm(range(0, len(indices), batch_size)):
                waveforms = f["waveforms"][:, :, indices[i : i + batch_size]].transpose(2, 0, 1)
                waveforms = np.nan_to_num(waveforms)
                f_new["waveforms"][i : i + batch_size] = waveforms
