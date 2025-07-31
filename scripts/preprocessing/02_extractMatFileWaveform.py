"""
hdf5_seisbench_processor.py

A module to read processed earthquake data from a single HDF5 file and create SeisBench datasets.
Adapted from the original mat_reader.py to work with consolidated HDF5 files.
Memory-efficient version that writes data immediately instead of accumulating in memory.
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
import gc  # For garbage collection


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

    def read_group_recursive(self, group):
        """
        Recursively read HDF5 group into nested dictionary, handling both groups and datasets.

        Parameters:
        -----------
        group : h5py.Group
            The HDF5 group to read

        Returns:
        --------
        dict: Nested dictionary containing all data from the group
        """
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # It's a group, recurse into it
                result[key] = self.read_group_recursive(item)
            else:
                # It's a dataset, read it
                result[key] = self.read_dataset_safe(item)
        return result

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
            def __init__(self, group, reader):
                self.group = group
                self.reader = reader

                # Read basic earthquake info
                if "name" in group:
                    self.name = self.reader.read_dataset_safe(group["name"])
                else:
                    self.name = eq_group_name

                # Read GAN data
                self.gan = self._read_structure_data(group.get("gan"))

                # Read recs data
                self.recs = self._read_structure_data(group.get("recs"))

            def _read_structure_data(self, data_group):
                """Read a structure (group or dict) recursively."""
                class StructData:
                    def __init__(self, data_dict):
                        if data_dict:
                            for key, value in data_dict.items():
                                setattr(self, key, value)

                if data_group is None:
                    return StructData({})

                if isinstance(data_group, h5py.Group):
                    # It's an HDF5 group, read it recursively
                    data_dict = self.reader.read_group_recursive(data_group)
                    return StructData(data_dict)
                else:
                    # It might be already processed data
                    return StructData(data_group if isinstance(data_group, dict) else {})

        return EarthquakeData(eq_group, self)

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
            return self.read_group_recursive(params_group)

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


def safe_access_array(data, index, default_value):
    """
    Safely access array element at index, with fallback to default value.

    Parameters:
    -----------
    data : array-like or scalar
        The data to access
    index : int
        Index to access
    default_value : any
        Default value if access fails

    Returns:
    --------
    The value at index or default_value
    """
    try:
        if np.isscalar(data):
            return data
        elif hasattr(data, '__len__') and len(data) > index:
            return data[index]
        elif hasattr(data, '__len__') and len(data) > 0:
            return data[0]  # Return first element if index out of bounds
        else:
            return default_value
    except:
        return default_value


if __name__ == "__main__":
    # Configuration
    hdf5_file_path = "/scratch/kpalgunadi/data/general/japan/bosai22/dl20220725/arx20220730/proj/wfGAN_python_hdf5/out/wfGAN_python_hdf5_processed_earthquakes.h5"

    print(f"Processing HDF5 file: {hdf5_file_path}")

    # Create dataset paths
    folder_path = "new_GM0_hdf5/"
    event_dir = Path(folder_path)
    obs_dir = event_dir / "observed_01_new"
    output_hdf5_path = obs_dir / "processed_waveforms.h5"

    # Create directories if they don't exist
    obs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output HDF5 path: {output_hdf5_path}")

    # Initialize counters and data collectors
    total_traces_written = 0
    earthquakes_processed = 0

    # Lists to collect all data
    all_waveforms = []
    all_trace_names = []
    all_trace_id_z = []
    all_trace_id_n = []
    all_trace_id_e = []
    all_trace_start_times = []
    all_trace_sampling_rates = []
    all_trace_npts = []
    all_station_longitudes = []
    all_station_latitudes = []
    all_station_network_codes = []
    all_station_codes = []
    all_trace_channels = []
    all_path_ep_distances = []
    all_vs30 = []
    all_azimuths = []
    all_back_azimuths = []

    # Event parameters
    all_source_ids = []
    all_source_magnitudes = []
    all_source_magnitude_types = []
    all_source_types = []
    all_source_latitudes = []
    all_source_longitudes = []
    all_source_depths = []
    all_source_origin_times = []
    all_source_strikes = []
    all_source_dips = []
    all_source_rakes = []
    all_azimuthal_gaps = []

    with HDF5EarthquakeReader(hdf5_file_path) as reader:
        # Get all earthquake groups
        earthquake_groups = reader.get_earthquake_groups()
        earthquake_groups.sort()  # Process in order

        print(f"Found {len(earthquake_groups)} earthquakes in HDF5 file")

        for eq_idx, eq_group_name in enumerate(earthquake_groups):
            print(f"Processing {eq_group_name} ({eq_idx + 1}/{len(earthquake_groups)})")

            try:
                eqs = reader.get_earthquake_data(eq_group_name)

                # Extract earthquake data with safer access
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

                # Safe access to filenames
                z_filenames = getattr(eqs.recs, 'z_filenames', [])
                n_filenames = getattr(eqs.recs, 'n_filenames', [])
                e_filenames = getattr(eqs.recs, 'e_filenames', [])

                try:
                    size_sta = len(sta_lat)
                except:
                    print("Number of station is only 1 or data is scalar")
                    if np.isscalar(sta_lat):
                        size_sta = 1
                        sta_lat = [sta_lat]
                        sta_lon = [sta_lon]
                    else:
                        continue

                # Handle scalar values that should be arrays
                if np.isscalar(source_lat):
                    source_lat = np.array([source_lat])
                    source_lon = np.array([source_lon])
                if np.isscalar(sta_lat):
                    sta_lat = np.array([sta_lat])
                    sta_lon = np.array([sta_lon])
                    size_sta = 1

                # Convert to numpy arrays for safe indexing
                sta_lat = np.array(sta_lat)
                sta_lon = np.array(sta_lon)
                source_lat = np.array(source_lat)
                source_lon = np.array(source_lon)

                hypo = (source_lat[0], source_lon[0])
                stations = np.vstack((sta_lat, sta_lon)).T
                azi_gap = calculate_azimuthal_gap(hypo, stations)

                traces_this_earthquake = 0

                for j in range(size_sta):
                    try:
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

                        # Calculate back azimuth
                        current_source_lat = safe_access_array(source_lat, j, source_lat[0])
                        current_source_lon = safe_access_array(source_lon, j, source_lon[0])
                        current_sta_lat = safe_access_array(sta_lat, j, 0)
                        current_sta_lon = safe_access_array(sta_lon, j, 0)

                        baz = gps2dist_azimuth(
                            current_source_lat,
                            current_source_lon,
                            current_sta_lat,
                            current_sta_lon,
                        )
                        theo_azi = np.round(baz[1], 2)
                        theo_back_azi = np.round(baz[2], 2)

                        # Store waveform data (transpose to [3, n_samples] format for NEZ)
                        all_waveforms.append(trace.T)  # Shape: [3, n_samples]

                        # Store trace parameters
                        all_trace_names.append(f"{eqs.name}_{str(t0)}_{safe_access_array(sta_network, j, 'UN')}_{safe_access_array(sta_name, j, 'UNKN')}")
                        all_trace_id_z.append(str(safe_access_array(z_filenames, j, "unknown_z")))
                        all_trace_id_n.append(str(safe_access_array(n_filenames, j, "unknown_n")))
                        all_trace_id_e.append(str(safe_access_array(e_filenames, j, "unknown_e")))
                        all_trace_start_times.append(str(t0))
                        all_trace_sampling_rates.append(100)
                        all_trace_npts.append(len(trace[:, 0]))
                        all_station_longitudes.append(current_sta_lon)
                        all_station_latitudes.append(current_sta_lat)
                        all_station_network_codes.append(str(safe_access_array(sta_network, j, "UN")))
                        all_station_codes.append(str(safe_access_array(sta_name, j, "UNKN")))
                        all_trace_channels.append("NEZ")
                        all_path_ep_distances.append(safe_access_array(rhyp, j, -999))
                        all_vs30.append(safe_access_array(vs30, j, -1))
                        all_azimuths.append(theo_azi)
                        all_back_azimuths.append(theo_back_azi)

                        # Store event parameters
                        all_source_ids.append(f"{eqs.name}_{str(t0)}")
                        all_source_magnitudes.append(safe_access_array(mag, j, mag[0] if hasattr(mag, '__len__') and len(mag) > 0 else -999))
                        all_source_magnitude_types.append("moment_magnitude")
                        all_source_types.append("global CMT")
                        all_source_latitudes.append(current_source_lat)
                        all_source_longitudes.append(current_source_lon)
                        all_source_depths.append(safe_access_array(source_dep, j, source_dep[0] if hasattr(source_dep, '__len__') and len(source_dep) > 0 else 0) * 1000.0)
                        all_source_origin_times.append(str(t0))
                        all_source_strikes.append(safe_access_array(source_strike, j, source_strike[0] if hasattr(source_strike, '__len__') and len(source_strike) > 0 else -999))
                        all_source_dips.append(safe_access_array(source_dip, j, source_dip[0] if hasattr(source_dip, '__len__') and len(source_dip) > 0 else -999))
                        all_source_rakes.append(safe_access_array(source_rake, j, source_rake[0] if hasattr(source_rake, '__len__') and len(source_rake) > 0 else -999))
                        all_azimuthal_gaps.append(azi_gap)

                        total_traces_written += 1
                        traces_this_earthquake += 1

                    except Exception as e:
                        print(f"Error processing station {j} in {eq_group_name}: {e}")
                        continue

                earthquakes_processed += 1
                print(f"  ✓ Processed {traces_this_earthquake} traces from {eq_group_name}")

                # Clear earthquake data from memory
                del eqs

            except Exception as e:
                print(f"Error processing {eq_group_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Convert lists to numpy arrays for HDF5 storage
    print(f"\nConverting {total_traces_written} traces to arrays...")

    # Convert waveforms to a single array
    # Find maximum length to pad shorter traces
    max_length = max(wf.shape[1] for wf in all_waveforms)
    waveforms_array = np.full((len(all_waveforms), 3, max_length), np.nan)

    for i, wf in enumerate(all_waveforms):
        waveforms_array[i, :, :wf.shape[1]] = wf

    # Convert string lists to byte arrays for HDF5
    def string_list_to_bytes(str_list):
        return [s.encode('utf-8') for s in str_list]

    # Save to HDF5
    print(f"Saving to HDF5 file: {output_hdf5_path}")

    with h5py.File(output_hdf5_path, "w") as h5f:
        # Trace parameters
        h5f.create_dataset("trace_name", data=string_list_to_bytes(all_trace_names))
        h5f.create_dataset("trace_id_z", data=string_list_to_bytes(all_trace_id_z))
        h5f.create_dataset("trace_id_n", data=string_list_to_bytes(all_trace_id_n))
        h5f.create_dataset("trace_id_e", data=string_list_to_bytes(all_trace_id_e))
        h5f.create_dataset("trace_start_time", data=string_list_to_bytes(all_trace_start_times))
        h5f.create_dataset("trace_sampling_rate_hz", data=np.array(all_trace_sampling_rates))
        h5f.create_dataset("trace_npts", data=np.array(all_trace_npts))
        h5f.create_dataset("station_longitude_deg", data=np.array(all_station_longitudes))
        h5f.create_dataset("station_latitude_deg", data=np.array(all_station_latitudes))
        h5f.create_dataset("station_network_code", data=string_list_to_bytes(all_station_network_codes))
        h5f.create_dataset("station_code", data=string_list_to_bytes(all_station_codes))
        h5f.create_dataset("trace_channel", data=string_list_to_bytes(all_trace_channels))
        h5f.create_dataset("path_ep_distance_km", data=np.array(all_path_ep_distances))
        h5f.create_dataset("vs30", data=np.array(all_vs30))
        h5f.create_dataset("azimuth", data=np.array(all_azimuths))
        h5f.create_dataset("back_azimuth", data=np.array(all_back_azimuths))

        # Event parameters
        h5f.create_dataset("source_id", data=string_list_to_bytes(all_source_ids))
        h5f.create_dataset("source_magnitude", data=np.array(all_source_magnitudes))
        h5f.create_dataset("source_magnitude_type", data=string_list_to_bytes(all_source_magnitude_types))
        h5f.create_dataset("source_type", data=string_list_to_bytes(all_source_types))
        h5f.create_dataset("source_latitude_deg", data=np.array(all_source_latitudes))
        h5f.create_dataset("source_longitude_deg", data=np.array(all_source_longitudes))
        h5f.create_dataset("source_depth_m", data=np.array(all_source_depths))
        h5f.create_dataset("source_origin_time", data=string_list_to_bytes(all_source_origin_times))
        h5f.create_dataset("source_strike", data=np.array(all_source_strikes))
        h5f.create_dataset("source_dip", data=np.array(all_source_dips))
        h5f.create_dataset("source_rake", data=np.array(all_source_rakes))
        h5f.create_dataset("azimuthal_gap", data=np.array(all_azimuthal_gaps))

        # Waveform data
        h5f.create_dataset("waveforms", data=waveforms_array)

        # Add metadata
        h5f.attrs["total_traces"] = total_traces_written
        h5f.attrs["earthquakes_processed"] = earthquakes_processed
        h5f.attrs["component_order"] = "NEZ"
        h5f.attrs["waveform_shape"] = "n_traces x 3_components x n_samples"
        h5f.attrs["sampling_rate_hz"] = 100
        h5f.attrs["measurement"] = "acceleration"
        h5f.attrs["unit"] = "m/s2"
        h5f.attrs["instrument_response"] = "not restituted"
        h5f.attrs["created_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nHDF5 file '{output_hdf5_path}' created successfully!")
    print(f"Total earthquakes processed: {earthquakes_processed}")
    print(f"Total traces written: {total_traces_written}")
    print(f"Waveform array shape: {waveforms_array.shape}")
    print(f"Component order: NEZ")
    print(f"Data format: [n_traces, 3_components, n_samples]")

    # Optional: Print some statistics
    print(f"\nDataset statistics:")
    print(f"  Average trace length: {np.mean(all_trace_npts):.0f} samples")
    print(f"  Max trace length: {max(all_trace_npts)} samples")
    print(f"  Min trace length: {min(all_trace_npts)} samples")
    print(f"  Magnitude range: {np.min(all_source_magnitudes):.2f} - {np.max(all_source_magnitudes):.2f}")
    print(f"  Distance range: {np.min([d for d in all_path_ep_distances if d > 0]):.2f} - {np.max(all_path_ep_distances):.2f} km")
