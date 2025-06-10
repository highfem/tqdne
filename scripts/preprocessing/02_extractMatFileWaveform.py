"""
mat_reader.py

A module to read nested variables from a .mat (MATLAB) file using scipy.io.loadmat.
"""

import scipy.io
import seisbench.data as sbd
import seisbench.util as sbu
import numpy as np 
import pandas as pd 
from datetime import datetime
import glob
import subprocess
import math
from pathlib import Path
from obspy.geodetics.base import gps2dist_azimuth
from obspy import UTCDateTime, Stream, Trace

class NestedMatReader:
    """
    A reader class that loads a .mat file and provides methods to traverse
    nested dictionaries or MATLAB structures (struct arrays).
    """

    def __init__(self, file_path):
        """
        Initialize the reader by loading the .mat file.

        Parameters:
        -----------
        file_path : str
            Path to the .mat file to be read.
        """
        # struct_as_record=False and squeeze_me=True help in reading MATLAB
        # struct arrays in a more Python-friendly way.
        self.data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    def get_nested_value(self, data_structure, keys):
        """
        Recursively retrieve a value from a nested dictionary or MATLAB struct.

        Parameters:
        -----------
        data_structure : dict or MATLAB struct-like object
            The root structure from which we want to retrieve nested values.
        keys : list of str
            A list of keys/attribute names to traverse the structure. For
            example, ['my_struct', 'my_field'] would look inside data_structure
            for 'my_struct', then inside that result for 'my_field'.

        Returns:
        --------
        any
            The value found at the nested location.

        Raises:
        -------
        KeyError
            If the key does not exist in the structure.
        """
        if not keys:
            # If no more keys, return the current structure
            return data_structure
        
        key = keys[0]
        
        # If data_structure is a dictionary
        if isinstance(data_structure, dict):
            if key not in data_structure:
                raise KeyError(f"Key '{key}' not found in dictionary: {list(data_structure.keys())}")
            return self.get_nested_value(data_structure[key], keys[1:])
        
        # Otherwise, assume data_structure is a MATLAB struct-like object
        # (accessed as attributes, not dictionary keys).
        if not hasattr(data_structure, key):
            raise KeyError(f"Attribute '{key}' not found in struct: {dir(data_structure)}")
        return self.get_nested_value(getattr(data_structure, key), keys[1:])

    def get_root_keys(self):
        """
        Return the top-level keys of the loaded .mat file.

        Returns:
        --------
        list
            A list of top-level variable names in the MATLAB file.
        """
        # MATLAB .mat files often contain '__header__', '__version__', etc.
        # We typically filter those out. You can customize this as needed.
        return [key for key in self.data.keys() if not key.startswith('__')]


def calculate_azimuthal_gap(hypocenter, station_coords):
    """
    Calculates the azimuthal gap of an earthquake.
    
    The azimuthal gap is defined as the largest angular gap (in degrees)
    between any two consecutive station azimuths as seen from the hypocenter.
    This metric is often used in seismology to assess the station coverage quality.
    
    Parameters:
        hypocenter: A tuple (latitude, longitude) in degrees for the earthquake.
        station_coords: A list of tuples [(lat, lon), ...] for each station in degrees.
    
    Returns:
        The maximum azimuthal gap in degrees. If fewer than two stations are provided,
        the function returns None.
    
    References:
        - Shearer, P. M. (2009). Introduction to Seismology. Cambridge University Press.
    """
    hypo_lat, hypo_lon = hypocenter
    
    # Compute azimuths from the hypocenter to each station.
    azimuths = []
    for (st_lat, st_lon) in station_coords:
        azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, st_lat, st_lon)
        azimuths.append(azimuth[1])
    
    if len(azimuths) < 2:
        print("Not enough stations to calculate an azimuthal gap. Using azimuth only instead")
        return azimuth[1]
    
    # Sort the azimuth angles in ascending order.
    azimuths.sort()
    
    # Calculate gaps between successive azimuth angles.
    gaps = []
    for i in range(1, len(azimuths)):
        gap = azimuths[i] - azimuths[i - 1]
        gaps.append(gap)
    
    # Add the gap between the last and the first angle (wrap-around gap)
    wrap_gap = 360 - (azimuths[-1] - azimuths[0])
    gaps.append(wrap_gap)
    
    # The azimuthal gap is the maximum gap.
    max_gap = max(gaps)
    return max_gap

def linear_interpolate(signal):
    """
    Fills NaNs in the signal with a simple linear interpolation.
    """
    n = len(signal)
    indices = np.arange(n)
    valid = ~np.isnan(signal)
    # np.interp fills missing values at boundaries with the first/last valid value.
    return np.interp(indices, indices[valid], signal[valid])

def analyze_frequency(signal, fs):
    """
    Computes the FFT of the input signal and estimates the dominant frequency band.

    Parameters:
        signal (numpy.ndarray): A complete (NaN-free) time-domain signal.
        fs (float): Sampling frequency.

    Returns:
        freqs (numpy.ndarray): Frequency bins (both positive and negative).
        power (numpy.ndarray): Power spectrum (magnitude squared) for nonnegative frequencies.
        f_low (float): Lower bound of the dominant frequency band.
        f_high (float): Upper bound of the dominant frequency band.
    """
    n = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)

    # Only consider nonnegative frequencies for estimating the dominant band.
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    power = np.abs(spectrum[pos_mask])**2

    # Define a significance threshold (here 5% of the maximum power).
    threshold = 0.05 * np.max(power)
    significant = power > threshold

    f_low, f_high = 0.1, 50

    return freqs, power, f_low, f_high

def spectral_gap_fill(signal, fs, num_iters=100, tol=1e-4):
    """
    Reconstructs a seismic signal with gaps using an iterative, frequency-constrained method.

    This method first ensures that the number of valid (non-NaN) data points
    exceeds the number of missing ones. It then:
      1. Fills missing values via linear interpolation.
      2. Analyzes the frequency content to determine a dominant frequency band.
      3. Iteratively enforces a frequency constraint (retaining only the dominant band)
         and data consistency (keeping valid data unchanged).

    Parameters:
        signal (numpy.ndarray): 1D seismic signal with missing values (NaNs).
        fs (float): Sampling frequency.
        num_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Reconstructed signal with gaps filled.

    Raises:
        ValueError: If the number of valid samples is not greater than the missing ones.
    """
    n = len(signal)
    valid_mask = ~np.isnan(signal)
    num_valid = np.sum(valid_mask)
    num_missing = np.sum(~valid_mask)

    if num_valid <= num_missing:
        raise ValueError(
            f"Insufficient valid data points (valid={num_valid}, missing={num_missing}). "
            "The number of NaN values must be less than the number of non-NaN values."
        )

    # Step 1: Initial guess using linear interpolation.
    x = linear_interpolate(signal)

    # Step 2: Frequency analysis to estimate the dominant frequency band.
    freqs, power, f_low, f_high = analyze_frequency(x, fs)

    # Build a frequency mask: retain only Fourier components with absolute frequencies within [f_low, f_high].
    freq_vals = np.fft.fftfreq(n, d=1/fs)
    freq_mask = (np.abs(freq_vals) >= f_low) & (np.abs(freq_vals) <= f_high)

    # Iterative reconstruction.
    x_old = x.copy()
    for iteration in range(num_iters):
        # Compute FFT of the current estimate.
        X = np.fft.fft(x)
        # Inverse FFT to obtain an updated time-domain signal.
        X[~freq_mask] = 0
        x_new = np.fft.ifft(X).real
        # Enforce data consistency: keep the original valid samples.
        x_new[valid_mask] = signal[valid_mask]

        # Check for convergence.
        diff = np.linalg.norm(x_new - x_old)
        if diff < tol:
            print(f"Converged in {iteration + 1} iterations (diff={diff:.6f}).")
            return x_new

        x_old = x_new.copy()
        x = x_new.copy()

    print(f"Reached maximum iterations without full convergence (last diff={diff:.6f}).")
    return x

if __name__ == "__main__":
    # Example usage (replace 'example.mat' with your file)

    result = []

    data = glob.glob("/scratch/kpalgunadi/data/general/japan/bosai22/dl20220725/arx20220730/proj/wfGAN_python/out/*.mat")
    data = np.sort(data)
    for file in data:
        print(f"Processing {file}")
        reader = NestedMatReader(file)
        
        eqs = reader.data['eq']
        date_time_str = eqs.gan.t0
        # date_str, time_str = date_time_str.split(' ')
        # date_format = '%d-%b-%Y'
        # date_obj = datetime.strptime(date_str, date_format)
        # formatted_str = date_obj.strftime('%Y-%m-%d') + 'T' + time_str
        t0 = UTCDateTime(date_time_str)
        vs30 = eqs.gan.vs30
        snr = eqs.gan.snr
        rhyp = eqs.gan.rhyp
        mag = eqs.gan.mag
        source_lat = eqs.gan.lat
        source_lon = eqs.gan.lon
        source_dep = eqs.gan.dep
        sta_network = eqs.gan.sta_network
        sta_name = eqs.gan.sta_name
        sta_lat = eqs.gan.sta_lat
        sta_lon = eqs.gan.sta_lon
        sta_alt = eqs.gan.sta_alt
        is_shallow_crustal = eqs.gan.is_shallow_crustal
        source_strike = eqs.gan.strike
        source_dip = eqs.gan.dip
        source_rake = eqs.gan.rake
        wfMat = eqs.gan.wfMat
        channel = ['N', 'E', 'Z']
        z_filenames = eqs.recs.z_filenames
        n_filenames = eqs.recs.n_filenames
        e_filenames = eqs.recs.e_filenames

        try:
            size_sta = len(sta_lat)
        except:
            print("Number of station is only 1")
            continue
        
        hypo = (source_lat[0], source_lon[0])
        stations = np.vstack((sta_lat, sta_lon)).T
        azi_gap = calculate_azimuthal_gap(hypo, stations)
        
        st = Stream()
        for j in range(size_sta):
            trace = wfMat[:,j,:]
            num_valid_N = np.sum(~np.isnan(trace[:,0]))
            num_missing_N = np.sum(np.isnan(trace[:,0]))
            num_valid_E = np.sum(~np.isnan(trace[:,1]))
            num_missing_E = np.sum(np.isnan(trace[:,1]))
            num_valid_Z = np.sum(~np.isnan(trace[:,2]))
            num_missing_Z = np.sum(np.isnan(trace[:,2]))
            
            if num_missing_N > 0:
                if num_valid_N > num_missing_N:
                    trace[:,0] = spectral_gap_fill(trace[:,0], 100, num_iters=100, tol=1e-5)
                if num_valid_E > num_missing_E:
                    trace[:,1] = spectral_gap_fill(trace[:,1], 100, num_iters=100, tol=1e-5)
                if num_valid_Z > num_missing_Z:
                    trace[:,2] = spectral_gap_fill(trace[:,2], 100, num_iters=100, tol=1e-5)
                if (num_valid_N > num_missing_N) and (num_valid_E > num_missing_E) and (num_valid_Z > num_missing_Z):
                    continue

            trN = Trace(trace[0,:])
            trE = Trace(trace[1,:])
            trZ = Trace(trace[2,:])

            trN.stats.sampling_rate = 1/0.01
            trN.stats.delta = 0.01
            trN.stats.starttime = t0 
            trN.stats.network = sta_network[j]
            trN.stats.station = sta_name[j]
            trN.stats.channel = 'HHN'
            st += trN

            trE.stats.sampling_rate = 1/0.01
            trE.stats.delta = 0.01
            trE.stats.starttime = t0 
            trE.stats.network = sta_network[j]
            trE.stats.station = sta_name[j]
            trE.stats.channel = 'HHE'
            st += trE

            trZ.stats.sampling_rate = 1/0.01
            trZ.stats.delta = 0.01
            trZ.stats.starttime = t0 
            trZ.stats.network = sta_network[j]
            trZ.stats.station = sta_name[j]
            trZ.stats.channel = 'HHZ'
            st += trZ


            baz = gps2dist_azimuth(source_lat[j], source_lon[j], sta_lat[j], sta_lon[j])
            theo_azi = np.round(baz[1],2)
            theo_back_azi = np.round(baz[2],2)

            trace_params = {
                "trace_name": f"{eqs.name}_{str(t0)}_{sta_network[j]}_{sta_name[j]}",
                "trace_id_z": z_filenames[j],
                "trace_id_n": n_filenames[j],
                "trace_id_e": e_filenames[j],
                "trace_start_time": t0, 
                "trace_sampling_rate_hz": 100,
                "trace_npts": len(trace[:,0]),
                "station_longitude_deg": sta_lon[j],  # add this just for niceness of plot w/ seisbench
                "station_latitude_deg": sta_lat[j],
                "station_network_code": sta_network[j],
                "station_code": sta_name[j],
                "trace_channel": 'NEZ',
                "path_ep_distance_km": rhyp[j],
                "azimuth": theo_azi,
                "back_azimuth": theo_back_azi,
                "vs30": vs30[j],
            }

            event_params = {
                "source_id": f"{eqs.name}_{str(t0)}",
                "source_magnitude": mag[j],
                "source_magnitude_type": "moment_magnitude",
                "source_type": "global CMT",
                "source_latitude_deg": source_lat[j],
                "source_longitude_deg": source_lon[j],
                "source_depth_m": source_dep[j] * 1000.0,
                "source_origin_time": t0,
                "source_strike": source_strike[j],
                "source_dip": source_dip[j],
                "source_rake": source_rake[j],
                "azimuthal_gap": azi_gap,
            }
            
            data_obs = sbu.stream_to_array(st, component_order="NEZ")[1]

            result.append(
                    {
                        "event_params": event_params,
                        "trace_params_obs": trace_params,
                        "data_obs": data_obs,
                    }
                )
    
    # create dataset
    folder_path = 'new_GM0/'
    event_dir = Path(folder_path)
    obs_dir = event_dir / "observed_01_new"
    metadata_path_obs = obs_dir / "metadata.csv"
    waveform_path_obs = obs_dir / "waveforms.hdf5"

    with sbd.WaveformDataWriter(metadata_path_obs, waveform_path_obs) as writer_obs:
        writer_obs.data_format = {
                    "dimension_order": "CW",
                    "component_order": "NEZ",
                    "measurement": "acceleration",
                    "unit": "m/s2",
                    "instrument_response": "restituted",
                }
        for station in result:
            writer_obs.add_trace(
                {
                    **station["event_params"],
                    **station["trace_params_obs"],
                },
                station["data_obs"],
            )
            writer_obs.flush_hdf5()

