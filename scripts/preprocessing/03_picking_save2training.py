import h5py
import numpy as np
import osmnx as ox
import seisbench.models as sbm
import seisbench
from obspy import Stream, Trace, UTCDateTime
from shapely.geometry import Point
import pandas as pd

# matplotlib.use("Tkagg")

# Print SeisBench version for reference (only using models)
print(f"Using SeisBench models version: {seisbench.__version__}")


def create_trace(data, stats_dict, channel):
    """
    Create an ObsPy Trace object from a 1D numpy array,
    fill NaNs with zeros, and update stats.
    """
    tr = Trace()
    tr.data = data
    tr.stats.update(stats_dict)
    tr.stats.channel = channel

    # Replace NaNs with zeros
    nan_idx = np.isnan(tr.data)
    tr.data[nan_idx] = 0

    return tr


def decode_bytes_array(byte_array):
    """
    Decode bytes array from HDF5 to string array.
    """
    return [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in byte_array]


class HDF5WaveformDataset:
    """
    A class to mimic SeisBench WaveformDataset functionality
    but reading from our custom HDF5 file format.
    """

    def __init__(self, hdf5_file_path):
        self.hdf5_file_path = hdf5_file_path
        self.h5f = None
        self.metadata = None
        self._load_data()

    def _load_data(self):
        """Load data from HDF5 file."""
        self.h5f = h5py.File(self.hdf5_file_path, 'r')

        # Create metadata DataFrame similar to SeisBench format
        metadata_dict = {}

        # Read all datasets except waveforms
        for key in self.h5f.keys():
            if key != 'waveforms':
                data = self.h5f[key][:]
                # Handle string/bytes data
                if isinstance(data[0], bytes):
                    data = decode_bytes_array(data)
                metadata_dict[key] = data

        # Add index column
        metadata_dict['index'] = np.arange(len(metadata_dict['source_id']))

        self.metadata = pd.DataFrame(metadata_dict)
        print(f"Loaded {len(self.metadata)} traces from HDF5 file")

    def get_waveforms(self, idx):
        """
        Get waveform data for a specific index.
        Returns data in shape [3, n_samples] (NEZ order).
        """
        if self.h5f is None:
            self._load_data()

        waveform_data = self.h5f['waveforms'][idx]  # Shape: [3, n_samples]
        return waveform_data

    def close(self):
        """Close HDF5 file."""
        if self.h5f:
            self.h5f.close()
            self.h5f = None

    def __del__(self):
        self.close()


# Get Japan polygon for onshore/offshore classification
japan_gdf = ox.geocode_to_gdf("Japan")
if japan_gdf.empty:
    raise ValueError("Could not obtain Japan's boundary from OSMnx.")

# Extract the geometry (Polygon or MultiPolygon) for Japan.
japan_polygon = japan_gdf.iloc[0].geometry


def classify_hypocenter(lat_arr, lon_arr, polygon):
    """
    Classify each earthquake hypocenter based on whether its point lies within the provided polygon.

    Args:
        lat_arr (array-like): Array of latitude values.
        lon_arr (array-like): Array of longitude values.
        polygon (shapely.geometry.Polygon or MultiPolygon): The boundary polygon.

    Returns:
        list: A list of classifications ('inside' or 'outside') for each coordinate pair.
    """
    classifications = []
    # Iterate over paired latitude and longitude values.
    for lat, lon in zip(lat_arr, lon_arr):
        # Create a point. Note: Shapely expects (longitude, latitude) as (x, y).
        point = Point(lon, lat)
        classifications.append("inside" if polygon.contains(point) else "outside")
    return classifications


# Load models
model = sbm.PhaseNet.from_pretrained("jma")

# Initialize lists for output data
event_ID_list = []
hypocentral_distance_list = []
hypocentre_depth_list = []
hypocentre_latitude_list = []
hypocentre_longitude_list = []
hypocentre_strike_list = []
hypocentre_dip_list = []
hypocentre_rake_list = []
is_onshore_list = []
magnitude_list = []
station_altitude_list = []
station_latitude_list = []
station_longitude_list = []
station_name_list = []
station_network_list = []
azimuth_list = []
back_azimuth_list = []
trace_sampling_rate_list = []
vs30_list = []
azimuthal_gap_list = []
waveform_list = []
z_filename_list = []

# Load data from HDF5 file
hdf5_file_path = "new_GM0_hdf5/observed_01_new/processed_waveforms.h5"  # Adjust path as needed
data_sample = HDF5WaveformDataset(hdf5_file_path)
metadata = data_sample.metadata

# Get unique event IDs
uniq_id = np.unique(metadata["source_id"])
iter = 0

for num_event in range(len(uniq_id)):
    print(f"{num_event + 1}/{len(uniq_id)}: Processing {uniq_id[num_event]}")
    data_filter = metadata[metadata.source_id == uniq_id[num_event]]
    id_filt = data_filter["index"].values
    sta_lat = metadata.station_latitude_deg[id_filt]
    sta_lon = metadata.station_longitude_deg[id_filt]

    # Build the Stream object
    st = Stream()
    for idx in data_filter["index"]:
        # Extract the 3-component waveform data (N, E, Z)
        data_array = data_sample.get_waveforms(idx)  # Shape: [3, n_samples]

        # Check for NaN values
        if (
            (np.isnan(np.mean(data_array[0, :])) == 1)
            or (np.isnan(np.mean(data_array[1, :])) == 1)
            or (np.isnan(np.mean(data_array[2, :])) == 1)
        ):
            print("there is nan")
            continue

        # Shared trace stats
        # Convert start time string to UTCDateTime
        start_time_str = metadata.loc[metadata.index[idx], "trace_start_time"]
        if isinstance(start_time_str, bytes):
            start_time_str = start_time_str.decode('utf-8')

        stats_dict = {
            "sampling_rate": metadata.loc[metadata.index[idx], "trace_sampling_rate_hz"],
            "delta": 1.0 / metadata.loc[metadata.index[idx], "trace_sampling_rate_hz"],
            "starttime": UTCDateTime(start_time_str),
            "network": metadata.loc[metadata.index[idx], "station_network_code"],
            "station": metadata.loc[metadata.index[idx], "station_code"],
        }

        # Channels for NEZ and their respective slices in data_array
        channel_map = {
            "HNN": data_array[0, :],
            "HNE": data_array[1, :],
            "HNZ": data_array[2, :],
        }

        for channel, waveform in channel_map.items():
            st += create_trace(waveform, stats_dict.copy(), channel)

        # Collect metadata
        event_ID_list.append(uniq_id[num_event])
        hypocentral_distance_list.append(metadata.loc[metadata.index[idx], "path_ep_distance_km"] * 1e3)
        hypocentre_depth_list.append(metadata.loc[metadata.index[idx], "source_depth_m"] * 1e-3)
        hypocentre_latitude_list.append(metadata.loc[metadata.index[idx], "source_latitude_deg"])
        hypocentre_longitude_list.append(metadata.loc[metadata.index[idx], "source_longitude_deg"])

        # evaluate hypocenter position based on onshore or off-shore
        is_onshore = classify_hypocenter(
            [float(metadata.loc[metadata.index[idx], "source_latitude_deg"])],
            [float(metadata.loc[metadata.index[idx], "source_longitude_deg"])],
            japan_polygon,
        )
        if is_onshore[0] == "inside":
            is_onshore = 1
        else:
            is_onshore = 0
        is_onshore_list.append(is_onshore)

        magnitude_list.append(metadata.loc[metadata.index[idx], "source_magnitude"])
        station_latitude_list.append(metadata.loc[metadata.index[idx], "station_latitude_deg"])
        station_longitude_list.append(metadata.loc[metadata.index[idx], "station_longitude_deg"])
        station_name_list.append(metadata.loc[metadata.index[idx], "station_code"])
        station_network_list.append(metadata.loc[metadata.index[idx], "station_network_code"])
        azimuth_list.append(metadata.loc[metadata.index[idx], "azimuth"])
        back_azimuth_list.append(metadata.loc[metadata.index[idx], "back_azimuth"])
        hypocentre_strike_list.append(metadata.loc[metadata.index[idx], "source_strike"])
        hypocentre_dip_list.append(metadata.loc[metadata.index[idx], "source_dip"])
        hypocentre_rake_list.append(metadata.loc[metadata.index[idx], "source_rake"])

        # Handle trace_id_z which might be bytes
        z_filename = metadata.loc[metadata.index[idx], "trace_id_z"]
        if isinstance(z_filename, bytes):
            z_filename = z_filename.decode('utf-8')
        z_filename_list.append(z_filename)

        azimuthal_gap_list.append(metadata.loc[metadata.index[idx], "azimuthal_gap"])
        trace_sampling_rate_list.append(metadata.loc[metadata.index[idx], "trace_sampling_rate_hz"])
        vs30_list.append(metadata.loc[metadata.index[idx], "vs30"])

    # Classify with PhaseNet, grouping traces in triplets (N, E, Z)
    pn_predictor = []
    pn_P = []
    pn_S = []
    pn_peak_time_P = []
    pn_peak_time_S = []
    pn_trace_id = []
    pn_peak_value_P = []
    pn_peak_value_S = []
    iter = 0

    for u in range(0, len(st), 3):
        stream_3c = st[u : u + 3]  # 3-component slice
        # print(f"Processing trace triplet index: {u}")
        stream_3c.detrend("demean")
        stream_3c.detrend("linear")
        st_extract = stream_3c.copy()

        # filter based on PhaseNet documentation 0.1 - 30 Hz
        stream_3c.filter("bandpass", freqmin=0.1, freqmax=30, zerophase=False)

        # Record the predictor name
        pn_predictor.append("PhaseNet-jma")
        annotations = model.annotate(stream_3c)

        # Run classification
        output = model.classify(stream_3c, P_threshold=0.05, S_threshold=0.04)
        pn_picks = output.picks

        if pn_picks:
            # Grab trace ID from the first pick for labeling
            first_pick_dict = pn_picks[0].__dict__
            pn_trace_id.append(first_pick_dict["trace_id"])

            # Variables to check if we have appended a P and an S
            p_has_been_appended = False
            s_has_been_appended = False

            for pick in pn_picks:
                pick_info = pick.__dict__
                phase = pick_info["phase"]

                # If we find a P pick for the first time
                if phase == "P" and not p_has_been_appended:
                    pn_P.append("P")
                    pn_peak_time_P.append(pick_info["peak_time"])
                    pn_peak_value_P.append(pick_info["peak_value"])
                    p_has_been_appended = True

                # If we find an S pick for the first time
                elif phase == "S" and not s_has_been_appended:
                    pn_S.append("S")
                    pn_peak_time_S.append(pick_info["peak_time"])
                    pn_peak_value_S.append(pick_info["peak_value"])
                    s_has_been_appended = True

                # Handle the scenario where there is only one pick total:
                # If that pick is P only, append default S or vice versa.
                if phase == "P" and len(pn_picks) == 1:
                    pn_S.append("S")
                    pn_peak_time_S.append(np.nan)
                    pn_peak_value_S.append(np.nan)
                    s_has_been_appended = True
                elif phase == "S" and len(pn_picks) == 1:
                    pn_P.append("P")
                    pn_peak_time_P.append(np.nan)
                    pn_peak_value_P.append(np.nan)
                    p_has_been_appended = True

            # If we never found a P pick
            if not p_has_been_appended:
                pn_P.append("P")
                pn_peak_time_P.append(np.nan)
                pn_peak_value_P.append(np.nan)

            # If we never found an S pick
            if not s_has_been_appended:
                pn_S.append("S")
                pn_peak_time_S.append(np.nan)
                pn_peak_value_S.append(np.nan)

        else:
            # No picks found at all
            default_trace_id = f"{stream_3c[0].stats.network}.{stream_3c[0].stats.station}."
            pn_trace_id.append(default_trace_id)

            pn_P.append("P")
            pn_peak_time_P.append(np.nan)
            pn_peak_value_P.append(np.nan)

            pn_S.append("S")
            pn_peak_time_S.append(np.nan)
            pn_peak_value_S.append(np.nan)

        try:
            onset_p = pn_peak_time_P[iter] - stream_3c[2].stats.starttime
        except:
            # print(f"pn_peak_time_P is {pn_peak_time_P[iter]}")
            st_extract[0].data = np.zeros(len(st_extract[0].data))
            st_extract[1].data = np.zeros(len(st_extract[1].data))
            st_extract[2].data = np.zeros(len(st_extract[2].data))
            st_data = np.vstack([tr.data for tr in st_extract]).T
            waveform_list.append(st_data)

            iter += 1
            continue

        starttime = stream_3c[2].stats.starttime + onset_p - 5

        st_extract[0].stats.starttime = starttime
        st_extract[1].stats.starttime = starttime
        st_extract[2].stats.starttime = starttime

        check_correctness = int(pn_peak_time_P[iter] - stream_3c[2].stats.starttime) * 100
        ave_start = np.max(np.sqrt(st_extract[2].data[0:300] ** 2))

        if (ave_start > 1e-3) & (metadata.loc[metadata.index[idx], "source_magnitude"] < 6):
            iter += 1
            print("skip waveforms, Max amplitude > 1e-3")
            st_extract[0].data = np.zeros(len(st_extract[0].data))
            st_extract[1].data = np.zeros(len(st_extract[1].data))
            st_extract[2].data = np.zeros(len(st_extract[2].data))
            st_data = np.vstack([tr.data for tr in st_extract]).T
            waveform_list.append(st_data)

            continue

        dataN = np.zeros(len(st_extract[0].data))
        dataE = np.zeros(len(st_extract[1].data))
        dataZ = np.zeros(len(st_extract[2].data))
        sample_move = int((pn_peak_time_P[iter] - stream_3c[2].stats.starttime) * 100 - 500)

        if sample_move > 0:
            dataN[:-sample_move] = st_extract[0].data[sample_move:]
            dataE[:-sample_move] = st_extract[1].data[sample_move:]
            dataZ[:-sample_move] = st_extract[2].data[sample_move:]
        elif sample_move == 0:
            dataN = st_extract[0].data
            dataE = st_extract[1].data
            dataZ = st_extract[2].data
        else:
            sample_move = -sample_move
            dataN[sample_move:] = st_extract[0].data[:-sample_move]
            dataE[sample_move:] = st_extract[1].data[:-sample_move]
            dataZ[sample_move:] = st_extract[2].data[:-sample_move]

        st_extract[0].data = dataN
        st_extract[1].data = dataE
        st_extract[2].data = dataZ

        st_extract.rotate(method="NE->RT", back_azimuth=back_azimuth_list[iter])
        print(st_extract)

        st_data = np.vstack([tr.data for tr in st_extract]).T

        waveform_list.append(st_data)

        # # Uncomment for debugging plots
        # fig = plt.figure(figsize=(15, 10))
        # axs = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
        # for i in range(3):
        #     axs[0].plot(st_extract[i].times(), st_extract[i].data, label=st_extract[i].stats.channel)
        #     axs[0].axvline(pn_peak_time_P[iter] - st_extract[i].stats.starttime)
        #     axs[0].axvline(5, c = 'r', linestyle='--')

        # axs[0].legend()
        # axs[1].legend()
        # axs[0].set_ylabel(f"{iter+1}, Max amplitude {ave_start}")
        # axs[0].set_title(f"{iter+1}, Max amplitude {ave_start}")
        # plt.show()
        # fig.savefig(f"sanity_check_{iter+1}.png", dpi=200, bbox_inches="tight")

        iter += 1

# Close the HDF5 file
data_sample.close()

# Convert lists to numpy arrays
event_ID_arr = np.array(event_ID_list).astype("S42")
hypocentral_distance_arr = np.array(hypocentral_distance_list)
hypocentre_depth_arr = np.array(hypocentre_depth_list)
hypocentre_latitude_arr = np.array(hypocentre_latitude_list)
hypocentre_longitude_arr = np.array(hypocentre_longitude_list)
is_onshore_arr = np.array(is_onshore_list)
magnitude_arr = np.array(magnitude_list)
station_altitude_arr = np.array(station_altitude_list)
station_latitude_arr = np.array(station_latitude_list)
station_longitude_arr = np.array(station_longitude_list)
station_name_arr = np.array(station_name_list).astype("S42")
station_network_arr = np.array(station_network_list).astype("S42")
azimuth_arr = np.array(azimuth_list)
back_azimuth_arr = np.array(back_azimuth_list)
azimuthal_gap_arr = np.array(azimuthal_gap_list)
hypocentre_strike_arr = np.array(hypocentre_strike_list)
hypocentre_dip_arr = np.array(hypocentre_dip_list)
hypocentre_rake_arr = np.array(hypocentre_rake_list)
trace_sampling_rate_arr = np.array(trace_sampling_rate_list)
vs30_arr = np.array(vs30_list)
z_filename_arr = np.array(z_filename_list)
z_filename_arr = z_filename_arr.astype("S")

# Create waveform array
n_events = len(event_ID_arr)
max_samples = 12501
waveforms_arr = np.zeros((n_events, max_samples, 3), dtype=np.float32)
for i in range(n_events):
    n_samps = waveform_list[i].shape[0]
    waveforms_arr[i, :n_samps, :] = waveform_list[i]

# Save final results to HDF5
file_path = "raw_waveforms.h5"
with h5py.File(file_path, "w") as h5f:
    h5f.create_dataset("event_ID", data=event_ID_arr)
    h5f.create_dataset("hypocentral_distance", data=hypocentral_distance_arr * 1e-3)
    h5f.create_dataset("hypocentre_depth", data=hypocentre_depth_arr)
    h5f.create_dataset("hypocentre_latitude", data=hypocentre_latitude_arr)
    h5f.create_dataset("hypocentre_longitude", data=hypocentre_longitude_arr)
    h5f.create_dataset("is_onshore", data=is_onshore_arr)
    h5f.create_dataset("magnitude", data=magnitude_arr)
    h5f.create_dataset("station_altitude", data=station_altitude_arr)
    h5f.create_dataset("station_latitude", data=station_latitude_arr)
    h5f.create_dataset("station_longitude", data=station_longitude_arr)
    h5f.create_dataset("station_name", data=station_name_arr)
    h5f.create_dataset("station_network", data=station_network_arr)
    h5f.create_dataset("azimuth", data=azimuth_arr)
    h5f.create_dataset("back_azimuth", data=back_azimuth_arr)
    h5f.create_dataset("azimuthal_gap", data=azimuthal_gap_arr)
    h5f.create_dataset("hypocentre_strike", data=hypocentre_strike_arr)
    h5f.create_dataset("hypocentre_dip", data=hypocentre_dip_arr)
    h5f.create_dataset("hypocentre_rake", data=hypocentre_rake_arr)
    h5f.create_dataset("trace_sampling_rate", data=trace_sampling_rate_arr)
    h5f.create_dataset("vs30", data=vs30_arr)
    h5f.create_dataset("z_filename", data=z_filename_arr)
    h5f.create_dataset("waveforms", data=waveforms_arr)

print(f"\nHDF5 file '{file_path}' created successfully with {n_events} valid events.")
