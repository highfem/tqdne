import multiprocessing as mp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from example_GMM import calculate_gmfs_distance
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import resample
from tqdm import tqdm


class MatFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.mat_dict = None

    def read_mat_file(self):
        """
        Reads a MATLAB (.mat) file and stores its contents as a dictionary.
        """
        try:
            mat_contents = scipy.io.loadmat(self.file_path, struct_as_record=False, squeeze_me=True)
            self.mat_dict = self.mat_to_dict(mat_contents)
        except Exception as e:
            print(f"Error reading .mat file: {e}")

    def mat_to_dict(self, mat_obj):
        """
        Recursively converts a MATLAB structure to a nested dictionary.
        """
        if isinstance(mat_obj, dict):
            return {
                key: self.mat_to_dict(value)
                for key, value in mat_obj.items()
                if not (key.startswith("__") and key.endswith("__"))
            }
        elif isinstance(mat_obj, np.ndarray):
            if mat_obj.size == 1:
                return self.mat_to_dict(mat_obj.item())
            else:
                return [self.mat_to_dict(element) for element in mat_obj]
        elif hasattr(mat_obj, "_fieldnames"):
            return {
                field: self.mat_to_dict(getattr(mat_obj, field)) for field in mat_obj._fieldnames
            }
        else:
            return mat_obj

    def print_mat_structure(self, mat_obj=None, indent=0):
        """
        Recursively prints the structure of the MATLAB file contents.
        """
        if mat_obj is None:
            mat_obj = self.mat_dict

        if isinstance(mat_obj, dict):
            for key, value in mat_obj.items():
                print(" " * indent + f"{key}:")
                self.print_mat_structure(value, indent + 4)
        elif isinstance(mat_obj, np.ndarray):
            print(" " * indent + f"Array, Shape: {mat_obj.shape}, Dtype: {mat_obj.dtype}")
        elif hasattr(mat_obj, "_fieldnames"):
            print(" " * indent + "MATLAB Object")
            for field in mat_obj._fieldnames:
                print(" " * (indent + 4) + f"{field}:")
                self.print_mat_structure(getattr(mat_obj, field), indent + 8)
        else:
            print(" " * indent + f"Type: {type(mat_obj)}")

    def process_data(self):
        """
        Processes specific data fields from the MATLAB file.
        """
        if self.mat_dict is None:
            print("MAT file not read yet.")
            return None, None

        try:
            rhyp = np.array(self.mat_dict["eq"]["gan"]["rhyp"])
            vs30 = np.array(self.mat_dict["eq"]["gan"]["vs30"])
            idx = np.where((vs30 > 0) & (~np.isnan(vs30)))

            rhyp = rhyp[idx]
            vs30 = vs30[idx]

            wf = np.array(self.mat_dict["eq"]["gan"]["wfMat"])

            return rhyp, vs30, wf
        except KeyError as e:
            print(f"Key error: {e}")
            return None, None, None


def shakeMap_cscale(mmi=None):
    """
    Returns a new ShakeMap MMI colormap for any vector of mmi values.
    Without input arguments, it returns the standard scale colormap.

    :param mmi: List or array of MMI values.
    :return: A matplotlib colormap object.

    # Example usage:
    cscale = shakeMap_cscale()

    # Generate some example data
    np.random.seed(0)
    x = np.random.uniform(0, 10, 100)
    y = np.random.uniform(0, 10, 100)
    z = np.random.uniform(1, 10, 100)  # These will be the MMI values

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=z, cmap=cscale, edgecolor='k')
    plt.colorbar(sc, label='MMI', ticks=np.arange(1, 11))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with ShakeMap MMI Colormap')
    plt.grid(True)
    plt.show()
    """

    if mmi is None:
        mmi = np.linspace(1, 10, 256)

    num_colors = len(mmi)

    # The colors in the original color map are the colors of the 11 edges of
    # the 10 bins, evenly spaced from 0.5 to 10.5.
    color_map = (
        np.array(
            [
                [255, 255, 255],
                [191, 204, 255],
                [160, 230, 255],
                [128, 255, 255],
                [122, 255, 147],
                [255, 255, 0],
                [255, 200, 0],
                [255, 145, 0],
                [255, 0, 0],
                [200, 0, 0],
                [128, 0, 0],
            ]
        )
        / 255.0
    )

    mmi_values = np.arange(1, 12)

    colormap_data = np.zeros((num_colors, 3))
    for i in range(3):
        interpolator = interp1d(mmi_values, color_map[:, i], kind="linear")
        colormap_data[:, i] = interpolator(mmi)

    # Create a colormap
    colormap = LinearSegmentedColormap.from_list("ShakeMapMMI", colormap_data, N=num_colors)

    return colormap


def pga_to_mmi(pga, unit="g"):
    """
    Convert Peak Ground Acceleration (PGA) to Modified Mercalli Intensity (MMI).

    Parameters:
    pga (float or numpy array): Peak Ground Acceleration.
    unit (str): Unit of the PGA ('g' for gravity or 'm/s^2' for meters per second squared).

    Returns:
    float or numpy array: Modified Mercalli Intensity (MMI).
    """
    # Ensure PGA is in the form of a numpy array for consistent operations
    pga = np.asarray(pga)

    # Conversion factor from m/s^2 to g
    if unit == "m/s^2":
        pga = pga / 9.80665  # 1 g = 9.80665 m/s^2
    elif unit == "cm/s^2":
        pga = pga / 9.80665 * 1e-2  # 1 g = 9.80665 m/s^2

    # Apply the empirical formula
    mmi = 3.66 * np.log10(pga) + 1.66

    return mmi


def calculate_gmrotd50(component1, component2):
    """
    Calculate the GMRotD50 from two horizontal component seismograms.

    Parameters:
    component1 (np.ndarray): Seismogram of the first horizontal component.
    component2 (np.ndarray): Seismogram of the second horizontal component.

    Returns:
    gmrotd50 (np.ndarray): The GMRotD50 values.
    """
    len1 = len(component1)
    len2 = len(component2)

    if len1 != len2:
        # Resample the shorter seismogram to match the length of the longer one
        if len1 < len2:
            component1 = resample(component1, len2)
        else:
            component2 = resample(component2, len1)

    # Number of rotation angles
    num_angles = 180
    gmrotd_values = np.zeros((num_angles, len(component1)))

    # Compute GMRotD for each rotation angle
    for angle in range(num_angles):
        theta = np.deg2rad(angle)
        rotated1 = component1 * np.cos(theta) + component2 * np.sin(theta)
        rotated2 = -component1 * np.sin(theta) + component2 * np.cos(theta)
        gmrotd = np.sqrt(rotated1**2 + rotated2**2)
        gmrotd_values[angle, :] = gmrotd

    # Compute GMRotD50 as the 50th percentile of the geometric mean
    gmrotd50 = np.percentile(gmrotd_values, 50, axis=0)

    return np.max(gmrotd50)


def plot_seismic_waveforms(
    waveforms,
    azimuthal_gap=None,
    hypocentral_distance=None,
    hypocentre_depth=None,
    magnitude=None,
    vs30s=None,
    time_vector=None,
    station_names=None,
    figsize=None,
    save_path=None,
    normalize=True,
):
    """
    Plot seismic waveforms in radial, tangential, and vertical components.

    Parameters:
    -----------
    waveforms : numpy.ndarray
        Array of shape [n, 3, n_samples] containing waveform data for n stations.
        The 3 components are radial, tangential, and vertical respectively.
    azimuthal_gap : float or list or numpy.ndarray, optional
        Azimuthal gap values in degrees.
    hypocentral_distance : float or list or numpy.ndarray, optional
        Hypocentral distance values in km.
    hypocentre_depth : float or list or numpy.ndarray, optional
        Hypocentral depth values in km.
    magnitude : float or list or numpy.ndarray, optional
        Magnitude values.
    vs30s : float or list or numpy.ndarray, optional
        VS30 values in m/s.
    time_vector : numpy.ndarray, optional
        Time vector for x-axis. If None, will use sample indices.
    station_names : list, optional
        Names of stations for labeling plots.
    figsize : tuple, optional
        Figure size in inches. If None, automatically determined based on n_stations.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    normalize : bool, optional
        Whether to normalize each waveform to its maximum absolute amplitude.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : numpy.ndarray
        Array of axes objects.
    """
    # Get dimensions
    n_stations = waveforms.shape[0]
    n_components = waveforms.shape[1]  # Should be 3
    n_samples = waveforms.shape[2]

    # Create time vector if not provided
    if time_vector is None:
        time_vector = np.arange(n_samples)

    # Create station names if not provided
    if station_names is None:
        station_names = [f"Station {i+1}" for i in range(n_stations)]

    # Set figsize based on number of stations if not specified
    if figsize is None:
        figsize = (12, max(4, 2.5 * n_stations))

    # Normalize waveforms if requested
    if normalize:
        # Find max amplitude for each station across all components
        max_abs = np.max(np.abs(waveforms), axis=(1, 2), keepdims=True)
        # Avoid division by zero
        max_abs[max_abs == 0] = 1.0
        waveforms_plot = waveforms / max_abs
    else:
        waveforms_plot = waveforms

    # Function to check if a parameter is an array of correct length
    def is_station_array(param):
        return (
            param is not None and isinstance(param, (list, np.ndarray)) and len(param) == n_stations
        )

    # Check if parameters are per-station arrays
    params_per_station = any(
        is_station_array(p)
        for p in [azimuthal_gap, hypocentral_distance, hypocentre_depth, magnitude, vs30s]
        if p is not None
    )

    # Helper function to safely format values
    def safe_format(value, format_str=".1f"):
        try:
            # Try to convert to float first
            value_float = float(value)
            return f"{value_float:{format_str}}"
        except (ValueError, TypeError):
            # If conversion fails, return as string
            return str(value)

    # Component names
    components = ["Radial", "Tangential", "Vertical"]

    # Create figure and axes
    fig, axes = plt.subplots(n_stations, 3, figsize=figsize, sharex=True)

    # If there's only one station, reshape axes for consistent indexing
    if n_stations == 1:
        axes = axes.reshape(1, 3)

    # Plot waveforms
    for i in range(n_stations):
        for j in range(3):
            axes[i, j].plot(time_vector, waveforms_plot[i, j, :], "k-")

            # Set y-limits to make waveforms visible
            if normalize:
                axes[i, j].set_ylim(-1.1, 1.1)
                axes[i, j].set_yticks([-1, 0, 1])

            # Add component label to the top row
            if i == 0:
                axes[i, j].set_title(components[j])

            # Add station name to the first column
            if j == 0:
                axes[i, j].set_ylabel(station_names[i])

            # Add parameter text to the right of the third column
            if j == 2:
                param_text = ""

                # Add parameters based on whether they're per-station or global
                if params_per_station:
                    # Per-station parameters
                    if azimuthal_gap is not None and is_station_array(azimuthal_gap):
                        param_text += f"Gap: {safe_format(azimuthal_gap[i])}°\n"

                    if hypocentral_distance is not None and is_station_array(hypocentral_distance):
                        param_text += f"Dist: {safe_format(hypocentral_distance[i])} km\n"

                    if hypocentre_depth is not None and is_station_array(hypocentre_depth):
                        param_text += f"Depth: {safe_format(hypocentre_depth[i])} km\n"

                    if magnitude is not None and is_station_array(magnitude):
                        param_text += f"Mag: {safe_format(magnitude[i])}\n"

                    if vs30s is not None and is_station_array(vs30s):
                        param_text += f"VS30: {safe_format(vs30s[i])} m/s"
                else:
                    # Global parameters (show on each station)
                    if azimuthal_gap is not None:
                        param_text += f"Gap: {safe_format(azimuthal_gap)}°\n"

                    if hypocentral_distance is not None:
                        param_text += f"Dist: {safe_format(hypocentral_distance)} km\n"

                    if hypocentre_depth is not None:
                        param_text += f"Depth: {safe_format(hypocentre_depth)} km\n"

                    if magnitude is not None:
                        param_text += f"Mag: {safe_format(magnitude)}\n"

                    if vs30s is not None:
                        param_text += f"VS30: {safe_format(vs30s)} m/s"

                # Add the parameter text if we have any
                if param_text:
                    axes[i, j].text(
                        1.05,
                        0.5,
                        param_text,
                        transform=axes[i, j].transAxes,
                        verticalalignment="center",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )

    # Add x-label to bottom row
    for j in range(3):
        axes[-1, j].set_xlabel("Time")

    plt.tight_layout()

    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


class SeismicParameters:
    """
    Class for reading and accessing seismic parameters from H5 files
    using dot notation (params.hypocentral_distance, params.magnitude, etc.)
    """

    def __init__(self, file_path=None):
        """
        Initialize SeismicParameters object.

        Parameters:
        -----------
        file_path : str, optional
            Path to the H5 file. If provided, automatically loads data.
        """
        # Initialize basic attributes
        self._file_path = None

        # Load data if file path is provided
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        """
        Load parameters from an H5 file.

        Parameters:
        -----------
        file_path : str
            Path to the H5 file

        Returns:
        --------
        self
            For method chaining
        """
        self._file_path = file_path

        # Open the H5 file with locking=False
        with h5py.File(file_path, "r", locking=False) as file:
            # Map specific parameter names to more intuitive attributes
            parameter_mapping = {
                "hypocentral_distance": "hypocentral_distance",
                "magnitude": "magnitude",
                "vs30": "vs30s",
                "hypocentre_depth": "hypocentre_depth",
                "azimuthal_gap": "azimuthal_gap",
            }

            # Add mapped parameters as attributes
            for h5_name, attr_name in parameter_mapping.items():
                if h5_name in file:
                    setattr(self, attr_name, file[h5_name][:])

            # Add any other parameters as attributes with original names
            for key in file.keys():
                if key not in parameter_mapping:
                    setattr(self, key, file[key][:])

        return self

    def get_data_info(self):
        """
        Returns information about the loaded data.

        Returns:
        --------
        dict
            Dictionary with information about available parameters and their shapes
        """
        info = {"file_path": self._file_path, "parameters": {}}

        # Get attributes that are numpy arrays (parameters)
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                if isinstance(value, np.ndarray):
                    info["parameters"][attr] = {
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "min": float(np.min(value)) if value.size > 0 else None,
                        "max": float(np.max(value)) if value.size > 0 else None,
                    }

        return info

    def __repr__(self):
        """String representation of the SeismicParameters object"""
        if self._file_path:
            return f"SeismicParameters(file='{self._file_path.split('/')[-1]}')"
        else:
            return "SeismicParameters(file=None)"


def process_waveform(index, EW, NS):
    gmrot50 = calculate_gmrotd50(EW, NS)
    return index, gmrot50


def parallel_process_waveforms(waveforms_EW, waveforms_NS, n_workers=4):
    with mp.Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_waveform,
                    [
                        (i, waveforms_EW[i, :], waveforms_NS[i, :])
                        for i in range(len(waveforms_EW[:, 0]))
                    ],
                ),
                total=len(waveforms_EW[:, 0]),
            )
        )
    return sorted(results, key=lambda x: x[0])


def integrate_frequency_domain(signal, dt):
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)

    # Create highpass filter mask (frequencies above 0.1 Hz)
    highpass_mask = np.abs(freqs) >= 0.1

    # Apply highpass filter
    fft_signal = fft_signal * highpass_mask

    # Integrate by dividing by j*omega
    fft_signal[1:] = fft_signal[1:] / (1j * 2 * np.pi * freqs[1:])
    fft_signal[0] = 0  # handle the DC component separately

    integrated_signal = np.fft.ifft(fft_signal).real
    return integrated_signal


def filter_frequency_domain(signal, dt):
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)

    # Create highpass filter mask (frequencies above 0.1 Hz)
    highpass_mask = np.abs(freqs) >= 0.1

    # Apply highpass filter
    fft_signal = fft_signal * highpass_mask

    filtered_signal = np.fft.ifft(fft_signal).real
    return filtered_signal


def evaluate_ratio(target, predicted, dt=0.01, n_worker=4, evaluate_obs=True, PGV=True):
    """
    Evaluate the ratio between target and predicted waveforms.

    Parameters:
    -----------
    target : numpy.ndarray
        Target waveforms with shape (n_samples, 2, time_steps)
    predicted : numpy.ndarray
        Predicted waveforms with shape (n_samples, 2, time_steps)
    dt : float
        Time step for integration
    n_worker : int, optional
        Number of workers for parallel processing (default: 4)
    evaluate_obs : bool, optional
        Whether to evaluate observations (default: True)

    Returns:
    --------
    dict or numpy.ndarray
        If evaluate_obs=True, returns a dictionary with PGV values for both observations and predictions
        Otherwise, returns only the predictions PGV values
    """
    # Load waveforms in chunks
    chunk_size = 1000  # adjust based on memory constraints
    num_chunks = int(np.ceil(target.shape[0] / chunk_size))

    results_obs = []
    results_gm0 = []

    if PGV:
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, target.shape[0])

            waveforms_chunk = target[start_idx:end_idx]
            tqdne_chunk = predicted[start_idx:end_idx]

            wf_NS = waveforms_chunk[:, 0, :]
            wf_EW = waveforms_chunk[:, 1, :]
            st_tqdne_EW = tqdne_chunk[:, 1, :]
            st_tqdne_NS = tqdne_chunk[:, 0, :]

            print("Integrate waveforms in the frequency domain")
            integrated_st_tqdne_EW = np.array(
                [integrate_frequency_domain(wf, dt) for wf in st_tqdne_EW]
            )
            integrated_st_tqdne_NS = np.array(
                [integrate_frequency_domain(wf, dt) for wf in st_tqdne_NS]
            )

            if evaluate_obs:
                integrated_wf_EW = np.array([integrate_frequency_domain(wf, dt) for wf in wf_EW])
                integrated_wf_NS = np.array([integrate_frequency_domain(wf, dt) for wf in wf_NS])

                print("Processing observation...")
                chunk_results_obs = parallel_process_waveforms(
                    integrated_wf_EW, integrated_wf_NS, n_workers=n_worker
                )
                results_obs.extend(chunk_results_obs)

            print("Processing GM0...")
            chunk_results_gm0 = parallel_process_waveforms(
                integrated_st_tqdne_EW, integrated_st_tqdne_NS, n_workers=n_worker
            )
            results_gm0.extend(chunk_results_gm0)

        # Extract PGV geometric mean values
        PGV_geom_mean_gm0 = np.array([result[1] for result in results_gm0])

        if evaluate_obs:
            PGV_geom_mean_obs = np.array([result[1] for result in results_obs])
            results = {
                "PGV_geom_mean_obs": PGV_geom_mean_obs,
                "PGV_geom_mean_gwm": PGV_geom_mean_gm0,
            }
        else:
            results = PGV_geom_mean_gm0
    else:
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, target.shape[0])

            waveforms_chunk = target[start_idx:end_idx]
            tqdne_chunk = predicted[start_idx:end_idx]

            wf_NS = waveforms_chunk[:, 0, :]
            wf_EW = waveforms_chunk[:, 1, :]
            wf_EW = np.array([filter_frequency_domain(wf, dt) for wf in wf_EW])
            wf_NS = np.array([filter_frequency_domain(wf, dt) for wf in wf_NS])

            st_tqdne_NS = tqdne_chunk[:, 0, :]
            st_tqdne_EW = tqdne_chunk[:, 1, :]
            st_tqdne_EW = np.array([filter_frequency_domain(wf, dt) for wf in st_tqdne_EW])
            st_tqdne_NS = np.array([filter_frequency_domain(wf, dt) for wf in st_tqdne_NS])

            if evaluate_obs:
                print("Processing observation...")
                chunk_results_obs = parallel_process_waveforms(wf_EW, wf_NS, n_workers=n_worker)
                results_obs.extend(chunk_results_obs)

            print("Processing GM0...")
            chunk_results_gm0 = parallel_process_waveforms(
                st_tqdne_EW, st_tqdne_NS, n_workers=n_worker
            )
            results_gm0.extend(chunk_results_gm0)

        # Extract PGV geometric mean values
        PGA_geom_mean_gm0 = np.array([result[1] for result in results_gm0])

        if evaluate_obs:
            PGA_geom_mean_obs = np.array([result[1] for result in results_obs])
            results = {
                "PGA_geom_mean_obs": PGA_geom_mean_obs,
                "PGA_geom_mean_gwm": PGA_geom_mean_gm0,
            }
        else:
            results = PGA_geom_mean_gm0

    return results


def calculate_distance_binned_ratios(
    PGX_geom_mean_obs, PGX_geom_mean_gm0, hypocentral_distance, n_bins=50
):
    """
    Calculate distance-binned statistics of the logarithmic ratio between observed and
    predicted peak ground motion (PGX) values.

    Parameters:
    -----------
    PGX_geom_mean_obs : numpy.ndarray
        Observed geometric mean PGX values
    PGX_geom_mean_gm0 : numpy.ndarray
        Predicted geometric mean PGX values
    hypocentral_distance : numpy.ndarray
        Hypocentral distances (rhyp) corresponding to each PGX value
    n_bins : int, optional
        Number of distance bins to use (default: 50)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'bin_centers': Centers of each distance bin
        - 'median_ratios': Median log10 ratios in each bin
        - 'std_ratios': Standard deviations of log10 ratios in each bin
        - 'bin_edges': Edges of the distance bins
        - 'ratio_values': Original log10 ratio values
    """
    import numpy as np

    # Validate input arrays have the same length
    if not (len(PGX_geom_mean_obs) == len(PGX_geom_mean_gm0) == len(hypocentral_distance)):
        raise ValueError("Input arrays must have the same length")

    # Calculate log10 ratio of observed to predicted PGX
    ratio = np.log10((PGX_geom_mean_obs) / (PGX_geom_mean_gm0))

    # Create distance bins
    r_bin = np.linspace(min(hypocentral_distance), max(hypocentral_distance), n_bins)

    # Initialize result arrays
    r_b = []  # Bin centers
    median = []  # Median ratio in each bin
    std = []  # Standard deviation in each bin
    counts = []  # Number of samples in each bin

    # Process each distance bin
    for i in range(len(r_bin) - 1):
        # Find indices of samples in current distance bin
        bin_indices = np.where(
            (hypocentral_distance > r_bin[i]) & (hypocentral_distance <= r_bin[i + 1])
        )[0]

        # Calculate bin center
        bin_center = 0.5 * (r_bin[i + 1] + r_bin[i])
        r_b.append(bin_center)

        # Calculate statistics if samples exist in this bin
        if len(bin_indices) > 0:
            median.append(np.median(ratio[bin_indices]))
            std.append(np.std(ratio[bin_indices]))
            counts.append(len(bin_indices))
        else:
            # No samples in this bin
            median.append(np.nan)
            std.append(np.nan)
            counts.append(0)

    # Convert lists to numpy arrays
    r_b = np.array(r_b)
    median = np.array(median)
    std = np.array(std)
    counts = np.array(counts)

    # Return results as a dictionary
    return {
        "bin_centers": r_b,
        "median_ratios": median,
        "std_ratios": std,
        "bin_counts": counts,
        "bin_edges": r_bin,
        "ratio_values": ratio,
    }


def ratio_gmm_pgv(ave_magnitude, ave_vs30, depth):
    mag = ave_magnitude
    if mag > 7.5 and mag < 8.0:
        rupture_aratio = 4
    elif mag >= 8 and mag < 8.5:
        rupture_aratio = 8
    elif mag >= 8.5:
        rupture_aratio = 25
    else:
        rupture_aratio = 2
    strike = 236
    dip = 51
    rake = 110
    lon = 137.89
    lat = 36.69
    depth = depth
    Vs30 = ave_vs30
    hypocenter = [lon, lat, depth]
    imts = ["PGV"]
    gmpes = ["BooreEtAl2014", "Kanno2006Shallow"]

    gms, jb_distance = calculate_gmfs_distance(
        mag, rupture_aratio, strike, dip, rake, hypocenter, imts, Vs30, gmpes
    )

    return gms, jb_distance


def ratio_gmm_pga(ave_magnitude, ave_vs30, depth):
    mag = ave_magnitude
    if mag > 7.5 and mag < 8.0:
        rupture_aratio = 4
    elif mag >= 8 and mag < 8.5:
        rupture_aratio = 8
    elif mag >= 8.5:
        rupture_aratio = 25
    else:
        rupture_aratio = 2
    strike = 236
    dip = 51
    rake = 110
    lon = 137.89
    lat = 36.69
    depth = depth
    Vs30 = ave_vs30
    hypocenter = [lon, lat, depth]
    imts = ["PGA"]
    gmpes = ["BooreEtAl2014", "Kanno2006Shallow"]

    gms, jb_distance = calculate_gmfs_distance(
        mag, rupture_aratio, strike, dip, rake, hypocenter, imts, Vs30, gmpes
    )

    return gms, jb_distance


def highpass_filter(data, cutoff_freq=0.1, sampling_rate=100):
    """
    Apply a causal high-pass filter to a waveform with dimensions [n,3,num_sample]

    Parameters:
    -----------
    data : numpy.ndarray
        Input data with shape [n,3,num_sample]
    cutoff_freq : float, optional
        Cutoff frequency of the high-pass filter in Hz (default: 0.1)
    sampling_rate : float, optional
        Sampling rate of the waveform in Hz (default: 100, based on dt=0.01s)

    Returns:
    --------
    numpy.ndarray
        Filtered data with the same shape as input
    """
    # Get the dimensions of the input data
    n, channels, num_samples = data.shape

    # Design a high-pass filter
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype="high")

    # Initialize the output array with the same shape as input
    filtered_data = np.zeros_like(data)

    # Apply the causal filter to each waveform
    for i in range(n):
        for j in range(channels):
            filtered_data[i, j, :] = signal.lfilter(b, a, data[i, j, :])

    return filtered_data


def print_dataset_summary(dataset_raw):
    """
    Print a clean summary of dataset variable ranges.

    Parameters:
    - dataset_raw: Dataset object with hypocentral_distance, magnitude, azimuthal_gap, vs30s attributes
    """
    print("=" * 60)
    print("                    DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Number of Data':<25} = {len(dataset_raw.hypocentral_distance)}")
    print(f"{'Variable':<25} {'Min':<15} {'Max':<15} {'Unit':<10}")
    print("-" * 60)
    print(
        f"{'Hypocentral Distance':<25} {min(dataset_raw.hypocentral_distance):<15.2f} {max(dataset_raw.hypocentral_distance):<15.2f} {'km':<10}"
    )
    print(
        f"{'Magnitude':<25} {min(dataset_raw.magnitude):<15.2f} {max(dataset_raw.magnitude):<15.2f} {'':<10}"
    )
    print(
        f"{'Azimuthal Gap':<25} {min(dataset_raw.azimuthal_gap):<15.2f} {max(dataset_raw.azimuthal_gap):<15.2f} {'degrees':<10}"
    )
    print(
        f"{'Vs30':<25} {min(dataset_raw.vs30s):<15.2f} {max(dataset_raw.vs30s):<15.2f} {'m/s':<10}"
    )
    print(
        f"{'Hypocenter depth':<25} {min(dataset_raw.hypocentre_depth):<15.2f} {max(dataset_raw.hypocentre_depth):<15.2f} {'km':<10}"
    )
    print("=" * 60)


def compare_waveforms(data1, data2, sample_rate, labels=None):
    """
    Compare two seismic waveforms by plotting them overlapped

    Parameters:
    -----------
    data1 : numpy.ndarray
        First seismic data with shape [3, n_samples]
    data2 : numpy.ndarray
        Second seismic data with shape [3, n_samples]
    sample_rate : float
        Sampling rate in Hz
    labels : list, optional
        List of two strings for the legend labels
    """
    if labels is None:
        labels = ["Waveform 1", "Waveform 2"]

    # Ensure both datasets have the same length for time axis
    min_samples = min(data1.shape[1], data2.shape[1])
    data1 = data1[:, :min_samples]
    data2 = data2[:, :min_samples]

    # Generate time axis
    time = np.linspace(0, min_samples / sample_rate, min_samples)

    # Apply window for FFT (reshape for broadcasting with [3, n_samples])
    window = np.hanning(min_samples).reshape(1, -1)
    windowed_data1 = data1 * window
    windowed_data2 = data2 * window

    # Compute FFT along the time axis (axis=1)
    fft_result1 = np.fft.rfft(windowed_data1, axis=1)
    fft_result2 = np.fft.rfft(windowed_data2, axis=1)
    fft_freqs = np.fft.rfftfreq(min_samples, 1 / sample_rate)
    fft_magnitude1 = np.abs(fft_result1)
    fft_magnitude2 = np.abs(fft_result2)

    # Create plots
    fig = plt.figure(figsize=(15, 12))
    components = ["Radial", "Tranverse", "UD"]

    for i in range(3):
        # Time domain comparison
        plt.subplot(3, 2, 2 * i + 1)
        plt.plot(time, data1[i, :], "k-", linewidth=1.5, label=labels[0])
        plt.plot(time, data2[i, :], "r-", linewidth=1.0, label=labels[1])
        plt.title(f"{components[i]} Component")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m/$s^2$)")
        plt.grid(True)
        plt.legend()

        # Frequency domain comparison
        plt.subplot(3, 2, 2 * i + 2)
        plt.loglog(fft_freqs, fft_magnitude1[i, :], "k-", linewidth=1.5, label=labels[0])
        plt.loglog(fft_freqs, fft_magnitude2[i, :], "r-", linewidth=1.0, label=labels[1])
        plt.title(f"{components[i]} Component")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (m/s$^2$ Hz$^{-1}$)")
        plt.grid(True, which="both", ls="-")
        plt.grid(True, which="minor", ls="--", alpha=0.4)
        plt.legend()

    plt.tight_layout()
    plt.show()

    return fig, fft_freqs, (fft_magnitude1, fft_magnitude2)


def fft(signal, dt=0.01):
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)

    return freqs[: int(N / 2)], abs(fft_signal)[: int(N / 2)]


def moving_average_envelope_adaptive(
    waveform,
    window_size: int = 128,
    log_eps: float = 1e-6,
):
    """
    Moving‐average envelope that only updates when the amplitude at the
    current sample jumps to more than 5× the previous sample’s amplitude.
    Otherwise it holds the previous envelope value.

    Parameters
    ----------
    waveform : array‐like
        Input waveform data. Last axis is time.
    window_size : int
        Length of the sliding window.
    log_eps : float
        Small epsilon to avoid log(0) when converting to log‐amplitude.

    Returns
    -------
    ndarray
        The log‐amplitude envelope, same shape as `waveform`.
    """
    # work with absolute amplitude
    abs_w = np.abs(waveform)
    env = np.zeros_like(abs_w)

    def process_track(x):
        out = np.zeros_like(x)
        for i in range(len(x)):
            start = max(0, i - window_size + 1)
            if i > 0 and x[i] > 5 * x[i - 1]:
                # sudden jump: recompute moving average
                out[i] = x[start : i + 1].mean()
            else:
                # otherwise hold last envelope (or raw amplitude at i=0)
                out[i] = out[i - 1] if i > 0 else x[0]
        return out

    # apply per‐channel/track
    env = np.apply_along_axis(process_track, axis=-1, arr=abs_w)

    # return log‐amplitude
    return env + log_eps
