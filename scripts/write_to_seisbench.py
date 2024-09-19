import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import seisbench.data as sbd
import seisbench.util as sbu
from obspy import UTCDateTime, Stream, Trace
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import argparse

parser = argparse.ArgumentParser(description='Convert the generative waveform model to seisbench format for a particular event')
parser.add_argument('path_metadata', help="path to station metadata (*.csv), Assumes you have a stations.csv file with columns 'network', 'station_code', 'channel_name' 'latitude', 'longitude', 'hypocentral_distance'")
parser.add_argument('path_gwm', help='path to generative waveform model (*.h5)')
parser.add_argument('--origin_time', nargs=1, default=(["2024-02-01T00:00:00.0"]), help='origin time when the earthquake occurred')
parser.add_argument('--hypocenter', nargs=3, default=([0,0,10]), help='hypocenter coordinate for latitude (degree), longitude (degree), depth (km)')
parser.add_argument('--magnitude', nargs=1, default=([6]), help='earthquake magnitude')
parser.add_argument('--num_realizations', nargs=1, default=([10]), help='number of realizations')
parser.add_argument('--trace_sampling_rate', nargs=1, default=([100.0]), help='sampling rate of the training data default is 100 samples/second')
parser.add_argument('base_path', help='path to folder to save into seisbench format')
args = parser.parse_args()

# Load station metadata from CSV file
stations_df = pd.read_csv(args.path_metadata)  # Assumes you have a stations.csv file with columns 'network', 'station_code', 'latitude', 'longitude'
stations_df = stations_df.drop(['channel_name', 'vs30', 'depth', 'magnitude'], axis=1)
stations_df = stations_df.values
stations_df = stations_df.astype(object)

columns_of_interest = stations_df[:, [0, 1]]
structured_array = np.core.records.fromarrays(columns_of_interest.T)
unique_combinations, indices, counts = np.unique(structured_array, return_index=True, return_counts=True)
indices = np.sort(indices)
stations_df_filter = stations_df[indices,:]
stations_df = pd.DataFrame(stations_df_filter, columns=['Network', 'Station', 'Latitude', 'Longitude', 'Distance'])

# Load additional data from HDF5 file
dataset_path = args.path_gwm
files_path = h5py.File(dataset_path, 'r', locking=False)
hypocentral_distance = files_path['hypocentral_distance'][:]
magnitude = files_path['magnitude'][:]
vs30 = files_path['vs30'][:]
waveforms = files_path['waveforms'][:]
is_shallow_crustal = files_path['is_shallow_crustal'][:]

# Earthquake information 
eq_lat = args.hypocenter[0]
eq_lon = args.hypocenter[1]
depth = args.hypocenter[2]
magnitude = args.magnitude[0]

# Number of waveforms and stations
n = len(stations_df)
waveforms_per_station = args.num_realizations[0]

# Example input data
source_location = {"longitude": eq_lon, "latitude": eq_lat, "depth": depth}
source_origin_time = UTCDateTime(args.origin_time[0])
waveform_unit = "m/s^2"
instrument_response = "not restituted"
component_order = "NEZ"
trace_sampling_rate = args.trace_sampling_rate[0] 
locations = ["*"] * n
channels = ["BHN", "BHE", "BHZ"]
Vp = 6.0  # P-wave velocity in km/s

# Define the paths for metadata and waveforms
base_path = Path(args.base_path)
base_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
metadata_path = base_path / "metadata.csv"
waveforms_path = base_path / "waveforms.hdf5"

# Function to create event parameters
def get_event_params():
    event_params = {
        "source_id": "synthetic_event_1",
        "source_origin_time": str(source_origin_time),
        "source_origin_uncertainty_sec": None,
        "source_latitude_deg": source_location["latitude"],
        "source_latitude_uncertainty_km": None,
        "source_longitude_deg": source_location["longitude"],
        "source_longitude_uncertainty_km": None,
        "source_depth_km": source_location["depth"],
        "source_depth_uncertainty_km": None,
        "source_magnitude": magnitude,
        "source_magnitude_uncertainty": None,
        "source_magnitude_type": None,
        "source_magnitude_author": None,
        "split": "train"  # Assuming all data is for training
    }
    return event_params

# Function to create trace parameters
def get_trace_params(station, trace_start_time, vs30_value, hypocentral_distance_km, is_shallow_crustal_value):
    trace_params = {
        "station_network_code": station["Network"],
        "station_code": station["Station"],
        "station_location_code": "00",
        "station_latitude_deg": station["Latitude"],
        "station_longitude_deg": station["Longitude"],
        "station_vs30": vs30_value,
        "station_is_shallow_crustal": bool(is_shallow_crustal_value),
        "hypocentral_distance_km": hypocentral_distance_km,
        "trace_channel": component_order,
        "trace_sampling_rate_hz": trace_sampling_rate,
        "trace_start_time": str(trace_start_time)
    }
    return trace_params

# Convert waveform to obspy Stream
def create_obspy_stream(waveform, channels, trace_start_time, sampling_rate, station):
    traces = []
    for i, channel in enumerate(channels):
        trace = Trace(data=waveform[i, :], header={
            'network': station["Network"],
            'station': station["Station"],
            'channel': channel,
            'starttime': trace_start_time,
            'sampling_rate': sampling_rate
        })
        traces.append(trace)
    return Stream(traces)

# Function to pick the trace start time using recursive_sta_lta
def pick_trace_start_time(data, sampling_rate):
    nsta = int(2 * sampling_rate)
    nlta = int(5 * sampling_rate)
    cft = recursive_sta_lta(data, nsta, nlta)
    on_off = trigger_onset(cft, 1.5, 0.5)  # Adjust trigger_on and trigger_off values as needed
    if len(on_off) > 0:
        start_sample = on_off[0][0]
        return start_sample / sampling_rate
    else:
        return 0  # If no trigger is found, return start time as 0

# Write to SeisBench format using WaveformDataWriter
with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
    writer.data_format = {
        "dimension_order": "CW",
        "component_order": component_order,
        "measurement": "acceleration",
        "unit": waveform_unit,
        "instrument_response": instrument_response,
    }
    
    event_params = get_event_params()

    for i in range(n):
        station = stations_df.iloc[i]
        for j in range(waveforms_per_station):
            waveform_index = i * waveforms_per_station + j
            waveform = waveforms[waveform_index]
            vs30_value = vs30[waveform_index]
            hypocentral_distance_km = hypocentral_distance[waveform_index]
            is_shallow_crustal_value = is_shallow_crustal[waveform_index]

            # Convert waveform to obspy Stream
            stream = create_obspy_stream(waveform, channels, source_origin_time, trace_sampling_rate, station)

            # Pick the trace start time
            trace_start_sample = pick_trace_start_time(stream[0].data, trace_sampling_rate)

            # Calculate the start time relative to the source origin time
            travel_time = hypocentral_distance_km / Vp
            trace_start_time = source_origin_time + travel_time - trace_start_sample / trace_sampling_rate

            trace_params = get_trace_params(station, trace_start_time, vs30_value, hypocentral_distance_km, is_shallow_crustal_value)

            actual_t_start, data, _ = sbu.stream_to_array(
                stream,
                component_order=component_order
            )

            sample = (trace_start_time - actual_t_start) * trace_sampling_rate
            trace_params["trace_P1_arrival_sample"] = int(sample)
            trace_params["trace_P1_status"] = "manual"

            writer.add_trace({**event_params, **trace_params}, data)

print("Metadata and waveforms saved successfully.")
