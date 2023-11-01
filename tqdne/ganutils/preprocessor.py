#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd


class Preprocessor:
    """ Simple class to downsample a data set from a HDF5 file.

    If the dimension is 1, downsampling is done on the first dimension.
    If the dimension is 3, downsampling is done on the second dimension.
    Otherwise, no downsampling is performed.

    :param filename: Datafile name to read from
    :type filename: str
    :param outfile: Datafile name where the downsampled data is written to
    :type outfile: str
    :param burnin: Length of file at the beginning to be discarded (burn-in)
    :type burnin: float
    :param duration: Total length of file in seconds
    :type duration: float
    :param sample_rate: Number of samples per second
    :type sample_rate: float
    :param channels: Number of Channels (dimension) of data after downsampling. E.g. 1 or 3.
    :type channels: int
    """

    def __init__(
        self,
        filename: str,
        outfile: str,
        burnin: float,
        duration: float,
        sample_rate: float,
        channels: int,
        waveforms_file: str = 'thisquakedoesnotexist/data/japan/waveforms.npy',
        attributes_file: str = 'thisquakedoesnotexist/data/japan/attributes.csv',
    ):
        self.filename = filename
        self.outfile = outfile
        self.h5_file = h5py.File(self.filename, "r")
        self.data = h5py.File(self.outfile, "a")
        self.burnin = burnin
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.waveforms_file = waveforms_file
        self.attributes_file = attributes_file

    def __repr__(self) -> str:
        return f"Downsampler instance for file: {self.filename}"

    def downsample(self, factor: int, lower_bound: float, upper_bound: float):
        """downsample_by_factor downsample data by a factor (frequency) and a threshold (magnitude).

        Selects all data with magnitudes between the provided lower and upper bounds,
        sampled at the frequency of factor.

        :param factor: Factor by which the data is downsampled
        :type factor: int
        :param lower_bound: Lower threshold from above which data is drawn
        :type lower_bound: float
        :param upper_bound: Upper threshold from below which data is drawn
        :type threshold: float
        """

        magnitudes = self.h5_file["magnitude"][0]
        threshold_start = np.argmax(magnitudes > lower_bound)
        threshold_end = np.argmax(magnitudes > upper_bound)
        print(f"Startpoint: {threshold_start}\nEndpoint: {threshold_end}")

        for val in self.h5_file:
            ds_file = np.array(self.h5_file[val])
            dimension = len(self.h5_file[val].shape)

            begin = int(self.burnin * self.sample_rate * factor)
            """
            print("Selecting data from beginning location: ", begin)
            print(f"self.burnin: {self.burnin} \t self.sample_rate: {self.sample_rate} \t factor: {factor}.")
            print(f"self.sample_rate * factor: {self.sample_rate * factor}")
            """
            end = int(self.duration * self.sample_rate * factor + begin)

            # all other attributes
            if dimension == 1:
                print(f'Downsampling for variable {self.h5_file[val]}: [begin:end:factor]')
                self.data[val] = self.h5_file[val][begin:end:factor]
            # magnitudes
            elif dimension == 2:
                print(f'Downsampling for variable {self.h5_file[val]}: [:, threshold_start:threshold_end]')
                self.data[val] = self.h5_file[val][:, threshold_start:threshold_end]
            # waveforms
            elif dimension == 3:
                print(f"Data of wfs orig: {self.h5_file[val][:10, :10]}")
                if self.channels == 1:
                    print(f'Downsampling for variable {self.h5_file[val]}: [1, begin:end:factor, threshold_start:threshold_end]')
                    self.data[val] = \
                        np.array(self.h5_file[val][1, begin:end:factor, threshold_start:threshold_end]).transpose((1, 0))
                elif self.channels == 3:
                    print(f'Downsampling for variable {self.h5_file[val]}: [:, begin:end:factor, threshold_start:threshold_end]')
                    self.data[val] = \
                        np.array(self.h5_file[val][:, begin:end:factor, threshold_start:threshold_end]).transpose((2, 0, 1))
                else:
                    print("Unclear dimension (channel) for downsampling. Try 3 or 1.")
                
                print(f"Data of wfs after sampling: {self.data[val][:10, :10]}")
            else:
                print(f"Nothing to be done for {val}, skipping downsampling.")
                self.data[val] = self.h5_file[val][:, :]
                continue

            
            print(f"Downsampling for {val}:")
            print(f"Old dimensions of data: {self.h5_file[val].shape}")
            print(f"New dimension of data: {self.data[val].shape}")
    
    
    def compute_pga(self):
        wfs_p = np.abs(self.data['waveforms'])

        # Calculate the max amplitude
        if self.channels == 1:
            wa = np.nanmax(wfs_p, axis=1)
            self.data['pga_v'] = wa # .reshape(1, -1)
        if self.channels == 3:
            wa = np.nanmax(wfs_p, axis=2)
            wa = np.nanmax(wa, axis=1)
            self.data['pga_v'] = wa

        print(self.data.keys())


    def select_shallow_crustal(self):
        n_obs = np.array(self.data['waveforms']).shape[0]
        
        """
        if self.channels == 3:
            n_obs = np.array(self.data['waveforms']).shape[0]
        """
        pga_v = self.data['pga_v'][:] #.reshape(-1)
        """
        if self.channels == 3:
            pga_v = self.data['pga_v'][1:, 1]
        """

        # breakpoint()

        df = pd.DataFrame({
            'i_wf': np.arange(n_obs), # index for waveform
            'dist': self.data['hypocentral_distance'][0, :],
            'ev_dep': self.data['hypocentre_depth'][0, :],
            'hypocentre_latitude': self.data['hypocentre_latitude'][0, :],
            'hypocentre_longitude': self.data['hypocentre_longitude'][0, :],
            'is_shallow_crustal': self.data['is_shallow_crustal'][0, :],
            'log10snr': self.data['log10snr'][0, :],
            'mag': self.data['magnitude'][0, :],
            'vs30': self.data['vs30'][0, :],
            'pga_v': pga_v,
        })

        # select only shallow crustal events
        df = df.loc[(df['is_shallow_crustal'] != 0)].reset_index(drop=True)
        df_index = df['i_wf'].values

        self.attr_df = df
        print(f'Old dimension of waveforms: {self.data["waveforms"].shape}')
        self.waveforms = self.data['waveforms'][df_index, :]
        
        # breakpoint()
        # # normalize waveforms
        # if self.channels == 3:
        #     self.waveforms = (np.array(self.waveforms).transpose(2, 1, 0) / np.array(pga_v[df_index])).transpose(2, 1, 0)
        #     self.waveforms[:, 0, :] = pga_v[df_index]

        # if self.channels == 1:
        #     self.waveforms = (np.array(self.waveforms).transpose(1, 0) / np.array(pga_v[df_index])).transpose(1, 0)
        #     self.waveforms[:, 0] = pga_v[df_index]
        # breakpoint()
        self.waveforms = np.nan_to_num(self.waveforms, 0)

        # self.data['waveforms'] = self.waveforms

        print(f'Old dimension of waveforms: {self.waveforms.shape}')


    """
    def select_shallow_crustal(self):
        n_obs = np.array(self.data['waveforms']).shape[0] -1 
        
        pga_v = self.data['pga_v'][1:] #.reshape(-1)
        

        # breakpoint()

        df = pd.DataFrame({
            'i_wf': np.arange(n_obs), # index for waveform
            'dist': self.data['hypocentral_distance'][0, 1:],
            'ev_dep': self.data['hypocentre_depth'][0, 1:],
            'hypocentre_latitude': self.data['hypocentre_latitude'][0, 1:],
            'hypocentre_longitude': self.data['hypocentre_longitude'][0, 1:],
            'is_shallow_crustal': self.data['is_shallow_crustal'][0, 1:],
            'log10snr': self.data['log10snr'][0, 1:],
            'mag': self.data['magnitude'][0, 1:],
            'vs30': self.data['vs30'][0, 1:],
            'pga_v': pga_v,
        })

        # select only shallow crustal events
        df = df.loc[(df['is_shallow_crustal'] != 0)].reset_index(drop=True)
        df_index = df['i_wf'].values

        self.attr_df = df
        print(f'Old dimension of waveforms: {self.data["waveforms"].shape}')
        self.waveforms = self.data['waveforms'][df_index, :]
        
        # breakpoint()
        # normalize waveforms
        if self.channels == 3:
            self.waveforms = (np.array(self.waveforms).transpose(2, 1, 0) / np.array(pga_v[df_index])).transpose(2, 1, 0)
        if self.channels == 1:
            self.waveforms = (np.array(self.waveforms).transpose(1, 0) / np.array(pga_v[df_index])).transpose(1, 0)
        
        breakpoint()
        self.waveforms = np.nan_to_num(self.waveforms, 0)
        
        print(f'Old dimension of waveforms: {self.waveforms.shape}')
    """

    def save_to_files(self):        
        self.attr_df.to_csv(self.attributes_file, index=False)
        np.save(self.waveforms_file, self.waveforms)
