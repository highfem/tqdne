"""
hdf5_seisbench_processor.py

A module to read processed earthquake data from a single HDF5 file and create SeisBench datasets.
Adapted from the original mat_reader.py to work with consolidated HDF5 files.
"""

import h5py
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


class HDF5EarthquakeReader:
    """
    A reader class that loads processed earthquake data from a single HDF5 file
    and provides methods to access individual earthquake records.
    """

    def __init__(self, file_path):
        """
        Initialize the reader by opening the HDF5 file.

        Parameters:
        -----------
        file_path : str
            Path to the HDF5 file containing processed earthquake data.
        """
        self.file_path = file_path
        self.h5f = None
        self._open_file()

    def _open_file(self):
        """Open the HDF5 file."""
        try:
            self.h5f = h5py.File(self.file_path, "r")
        except Exception as e:
            raise IOError(f"Cannot open HDF5 file {self.file_path}: {e}")

    def close(self):
        """Close the HDF5 file."""
        if self.h5f:
            self.h5f.close()
            self.h5f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_earthquake_groups(self):
        """
        Get list of earthquake group names in the HDF5 file.

        Returns:
        --------
        list: List of earthquake group names
        """
        if not self.h5f:
            self._open_file()

        return [key for key in self.h5f.keys() if key.startswith("earthquake_")]

    def read_dataset_safe(self, dataset):
        """
        Safely read a dataset, handling different data types.

        Parameters:
        -----------
        dataset : h5py.Dataset
            The HDF5 dataset to read

        Returns:
        --------
        The dataset value with appropriate type conversion
        """
        data = dataset[()]

        if isinstance(data, bytes):
            return data.decode()
        elif hasattr(data, "dtype") and data.dtype.kind in ["U", "S"]:
            if data.size == 1:
                return str(data.item())
            else:
                return [str(item) for item in data]
        else:
            return data

    def get_earthquake_data(self, eq_group_name):
        """
        Read complete earthquake data for a specific earthquake group.

        Parameters:
        -----------
        eq_group_name : str
            Name of the earthquake group

        Returns:
        --------
        dict: Dictionary containing all earthquake data
        """
        if not self.h5f:
            self._open_file()

        if eq_group_name not in self.h5f:
            raise KeyError(f"Earthquake group '{eq_group_name}' not found in HDF5 file")

        eq_group = self.h5f[eq_group_name]

        # Create a structured object to mimic the original MATLAB struct access pattern
        class EarthquakeData:
            def __init__(self, group):
                self.group = group

                # Read basic earthquake info
                if "name" in group:
                    self.name = self._read_safe(group["name"])
                else:
                    self.name = eq_group_name

                # Read GAN data
                self.gan = self._read_gan_data(group.get("gan", {}))

                # Read recs data
                self.recs = self._read_recs_data(group.get("recs", {}))

            def _read_safe(self, dataset):
                """Safely read dataset with type conversion."""
                data = dataset[()]
                if isinstance(data, bytes):
                    return data.decode()
                elif hasattr(data, "dtype") and data.dtype.kind in ["U", "S"]:
                    if data.size == 1:
                        return str(data.item())
                    else:
                        return [str(item) for item in data]
                else:
                    return data

            def _read_gan_data(self, gan_group):
                """Read GAN data structure."""

                class GANData:
                    pass

                gan_data = GANData()

                if not gan_group:
                    return gan_data

                # Read all GAN fields
                for key in gan_group.keys():
                    value = self._read_safe(gan_group[key])
                    setattr(gan_data, key, value)

                return gan_data

            def _read_recs_data(self, recs_group):
                """Read recs data structure."""

                class RecsData:
                    pass

                recs_data = RecsData()

                if not recs_group:
                    return recs_data

                # Read all recs fields
                for key in recs_group.keys():
                    value = self._read_safe(recs_group[key])
                    setattr(recs_data, key, value)

                return recs_data

        return EarthquakeData(eq_group)

    def get_processing_parameters(self):
        """
        Get the processing parameters from the HDF5 file.

        Returns:
        --------
        dict: Processing parameters used during earthquake processing
        """
        if not self.h5f:
            self._open_file()

        if "processing_parameters" in self.h5f:
            params_group = self.h5f["processing_parameters"]

            def read_group_recursive(group):
                result = {}
                for key in group.keys():
                    if isinstance(group[key], h5py.Group):
                        result[key] = read_group_recursive(group[key])
                    else:
                        result[key] = self.read_dataset_safe(group[key])
                return result

            return read_group_recursive(params_group)

        return {}


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
    for st_lat, st_lon in station_coords:
        azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, st_lat, st_lon)
        azimuths.append(azimuth[1])

    if len(azimuths) < 2:
        print("Not enough stations to calculate an azimuthal gap. Using azimuth only instead")
        return azimuth[1] if azimuths else 0

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
    freqs = np.fft.fftfreq(n, d=1 / fs)

    # Only consider nonnegative frequencies for estimating the dominant band.
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    power = np.abs(spectrum[pos_mask]) ** 2

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
    freq_vals = np.fft.fftfreq(n, d=1 / fs)
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
    # Configuration
    hdf5_file_path = "/scratch/kpalgunadi/data/general/japan/bosai22/dl20220725/arx20220730/proj/wfGAN_python_hdf5/out/wfGAN_python_hdf5_processed_earthquakes.h5"

    result = []

    print(f"Processing HDF5 file: {hdf5_file_path}")

    with HDF5EarthquakeReader(hdf5_file_path) as reader:
        # Get all earthquake groups
        earthquake_groups = reader.get_earthquake_groups()
        earthquake_groups.sort()  # Process in order

        print(f"Found {len(earthquake_groups)} earthquakes in HDF5 file")

        for eq_group_name in earthquake_groups:
            print(f"Processing {eq_group_name}")

            try:
                eqs = reader.get_earthquake_data(eq_group_name)

                # Extract earthquake data (same structure as original)
                date_time_str = eqs.gan.t0
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
                channel = ["N", "E", "Z"]
                z_filenames = eqs.recs.z_filenames
                n_filenames = eqs.recs.n_filenames
                e_filenames = eqs.recs.e_filenames

                try:
                    size_sta = len(sta_lat)
                except:
                    print("Number of station is only 1")
                    continue

                # Handle scalar values that should be arrays
                if np.isscalar(source_lat):
                    source_lat = np.array([source_lat])
                    source_lon = np.array([source_lon])
                if np.isscalar(sta_lat):
                    sta_lat = np.array([sta_lat])
                    sta_lon = np.array([sta_lon])
                    size_sta = 1

                hypo = (source_lat[0], source_lon[0])
                stations = np.vstack((sta_lat, sta_lon)).T
                azi_gap = calculate_azimuthal_gap(hypo, stations)

                st = Stream()
                for j in range(size_sta):
                    # Handle wfMat shape - it should be [3, n_stations, n_samples] -> [n_samples, n_stations, 3]
                    if wfMat.ndim == 3:
                        if wfMat.shape[0] == 3:  # [3, n_stations, n_samples]
                            trace = wfMat[:, j, :].T  # [n_samples, 3]
                        else:  # [n_samples, n_stations, 3]
                            trace = wfMat[:, j, :]
                    else:
                        print(f"Unexpected wfMat shape: {wfMat.shape}")
                        continue

                    num_valid_N = np.sum(~np.isnan(trace[:, 0]))
                    num_missing_N = np.sum(np.isnan(trace[:, 0]))
                    num_valid_E = np.sum(~np.isnan(trace[:, 1]))
                    num_missing_E = np.sum(np.isnan(trace[:, 1]))
                    num_valid_Z = np.sum(~np.isnan(trace[:, 2]))
                    num_missing_Z = np.sum(np.isnan(trace[:, 2]))

                    # Gap filling for missing data
                    if num_missing_N > 0:
                        if num_valid_N > num_missing_N:
                            trace[:, 0] = spectral_gap_fill(
                                trace[:, 0], 100, num_iters=100, tol=1e-5
                            )
                    if num_missing_E > 0:
                        if num_valid_E > num_missing_E:
                            trace[:, 1] = spectral_gap_fill(
                                trace[:, 1], 100, num_iters=100, tol=1e-5
                            )
                    if num_missing_Z > 0:
                        if num_valid_Z > num_missing_Z:
                            trace[:, 2] = spectral_gap_fill(
                                trace[:, 2], 100, num_iters=100, tol=1e-5
                            )

                    # Skip if all components have too much missing data
                    if (
                        (num_valid_N <= num_missing_N)
                        and (num_valid_E <= num_missing_E)
                        and (num_valid_Z <= num_missing_Z)
                    ):
                        continue

                    # Create ObsPy traces (note: wfMat order is N, E, Z)
                    trN = Trace(trace[:, 0])
                    trE = Trace(trace[:, 1])
                    trZ = Trace(trace[:, 2])

                    # Set trace statistics
                    for tr, comp in zip([trN, trE, trZ], ["HHN", "HHE", "HHZ"]):
                        tr.stats.sampling_rate = 1 / 0.01
                        tr.stats.delta = 0.01
                        tr.stats.starttime = t0
                        tr.stats.network = (
                            sta_network[j]
                            if hasattr(sta_network[j], "__len__")
                            else str(sta_network[j])
                        )
                        tr.stats.station = (
                            sta_name[j] if hasattr(sta_name[j], "__len__") else str(sta_name[j])
                        )
                        tr.stats.channel = comp
                        st += tr

                    # Calculate back azimuth
                    baz = gps2dist_azimuth(
                        source_lat[j] if j < len(source_lat) else source_lat[0],
                        source_lon[j] if j < len(source_lon) else source_lon[0],
                        sta_lat[j],
                        sta_lon[j],
                    )
                    theo_azi = np.round(baz[1], 2)
                    theo_back_azi = np.round(baz[2], 2)

                    trace_params = {
                        "trace_name": f"{eqs.name}_{str(t0)}_{sta_network[j]}_{sta_name[j]}",
                        "trace_id_z": z_filenames[j]
                        if hasattr(z_filenames[j], "__len__")
                        else str(z_filenames[j]),
                        "trace_id_n": n_filenames[j]
                        if hasattr(n_filenames[j], "__len__")
                        else str(n_filenames[j]),
                        "trace_id_e": e_filenames[j]
                        if hasattr(e_filenames[j], "__len__")
                        else str(e_filenames[j]),
                        "trace_start_time": t0,
                        "trace_sampling_rate_hz": 100,
                        "trace_npts": len(trace[:, 0]),
                        "station_longitude_deg": sta_lon[j],
                        "station_latitude_deg": sta_lat[j],
                        "station_network_code": sta_network[j]
                        if hasattr(sta_network[j], "__len__")
                        else str(sta_network[j]),
                        "station_code": sta_name[j]
                        if hasattr(sta_name[j], "__len__")
                        else str(sta_name[j]),
                        "trace_channel": "NEZ",
                        "path_ep_distance_km": rhyp[j],
                        "vs30": vs30[j],
                        "azimuth": theo_azi,
                        "back_azimuth": theo_back_azi,
                    }

                    event_params = {
                        "source_id": f"{eqs.name}_{str(t0)}",
                        "source_magnitude": mag[j] if j < len(mag) else mag[0],
                        "source_magnitude_type": "moment_magnitude",
                        "source_type": "global CMT",
                        "source_latitude_deg": source_lat[j]
                        if j < len(source_lat)
                        else source_lat[0],
                        "source_longitude_deg": source_lon[j]
                        if j < len(source_lon)
                        else source_lon[0],
                        "source_depth_m": (source_dep[j] if j < len(source_dep) else source_dep[0])
                        * 1000.0,
                        "source_origin_time": t0,
                        "source_strike": source_strike[j]
                        if j < len(source_strike)
                        else source_strike[0],
                        "source_dip": source_dip[j] if j < len(source_dip) else source_dip[0],
                        "source_rake": source_rake[j] if j < len(source_rake) else source_rake[0],
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

                    # Clear stream for next station
                    st.clear()

            except Exception as e:
                print(f"Error processing {eq_group_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Create dataset
    folder_path = "new_GM0_hdf5/"
    event_dir = Path(folder_path)
    obs_dir = event_dir / "observed_01_new"
    metadata_path_obs = obs_dir / "metadata.csv"
    waveform_path_obs = obs_dir / "waveforms.hdf5"

    # Create directories if they don't exist
    obs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating SeisBench dataset with {len(result)} records...")
    print(f"Metadata path: {metadata_path_obs}")
    print(f"Waveforms path: {waveform_path_obs}")

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

    print(f"Dataset creation completed! {len(result)} traces written to SeisBench format.")
