import sys
import numpy as np
from config import Config
from einops import rearrange 
from h5py import File
from tqdm import tqdm


def run(args):    
    config = Config(args.infile)
    with File(args.infile, "r") as f:
        # remove samples with vs30 <= 0        
        mask = f["vs30"][:] > 0        
        indices = np.arange(len(mask))[mask]
        with File(args.outfile, "w") as f_new:
            features = []            
            for key in config.features_keys:         
                print(key, "", f[key].shape)       
                feature = f[key][mask]                
                f_new.create_dataset(key, data=feature)
                features.append(feature)

            features = np.stack(features, axis=1)
            print("normalized features ", features.shape)
            normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
            f_new.create_dataset("normalized_features", data=normalized_features)

            _, t, channels = f["waveforms"].shape
            print("waveforms ", f["waveforms"].shape)

            f_new.create_dataset("waveforms", (len(indices), channels, t))
            batch_size = 1000
            for i in tqdm(range(0, len(indices), batch_size)):
                waveforms = f["waveforms"][indices[i : i + batch_size], ...]
                waveforms = rearrange(waveforms, "b t c -> b c t")
                waveforms = np.nan_to_num(waveforms)
                f_new["waveforms"][i : i + batch_size] = waveforms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Create a data set for training autoencoder and LDM models")
    parser.add_argument("--infile", type=str, default=None, help="Raw waveforms file, e.g., `raw_waveforms.h5`")
    parser.add_argument("--outfile", type=str, default=None, help="Result filename from calling this script,  e.g., `preprocessed_waveforms.h5`")
    args = parser.parse_args()
    if args.infile is None or args.outfile is None:
        parser.print_help()
        sys.exit(0)
    run(args)

