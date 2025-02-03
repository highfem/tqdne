"""
Script Name: create h5 for tqdne using STEAD dataset
Description:
    This script reads a CSV file and an HDF5 file containing seismic trace information. It writes relevant information into a new HDF5 file
    for tqdne structure.

"""

import pandas as pd
import h5py
import obspy
import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client


def make_stream(dataset):
    """
    Convert an HDF5 dataset of shape (samples, 3) into an ObsPy Stream object. 
    Inspired by https://github.com/smousavi05/STEAD

    Parameters
    ----------
    dataset : h5py.Dataset
        An HDF5 dataset that contains the seismic data with shape (n_samples, 3),
        where the columns typically represent the N, E, Z components.

    Returns
    -------
    obspy.Stream
        An ObsPy Stream containing three Traces (N, E, Z).
    """
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01  # Sample rate interval
    tr_E.stats.channel = dataset.attrs['receiver_type'] + 'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type'] + 'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type'] + 'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    # Create a Stream containing the three traces (N, E, Z)
    stream = obspy.Stream([tr_N, tr_E, tr_Z])

    return stream


def create_h5_file(file_path, df, dtfl):
    """
    Creates an HDF5 file in a STEAD-like structure and populates it with seismic metadata
    and waveforms. It processes each selected event by removing the instrument response,
    trimming the waveforms, and storing them in the new HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the output HDF5 file that will be created (e.g., 'stead_example.h5').
    df : pandas.DataFrame
        A filtered DataFrame containing event and station metadata.
    dtfl : h5py.File
        An open HDF5 file object containing the raw seismic data.

    Returns
    -------
    None
    """
    num_samples = len(df)

    # Extract metadata into numpy arrays
    event_ID = df.source_id.values
    hypocentral_distance = df.source_distance_km.values
    hypocentre_depth = df.source_depth_km.values
    hypocentre_latitude = df.source_latitude.values
    hypocentre_longitude = df.source_longitude.values
    is_shallow_crustal = (df['source_depth_km'] <= 25).astype(int)
    magnitude = df.source_magnitude.values
    station_altitude = df.receiver_elevation_m.values
    station_latitude = df.receiver_latitude.values
    station_longitude = df.receiver_longitude.values
    station_name = df.receiver_code.values
    station_network = df.network_code.values
    time_sample = df.p_arrival_sample.values
    time_arrival = df.p_travel_sec.values
    trace_start_time = df.trace_start_time.values

    # Randomly assigned vs30 values for demonstration, no vs30 value in STEAD dataset; adjust as necessary
    vs30 = np.random.randint(400, 1501, num_samples)

    # Prepare a list of dataset paths within the HDF5
    ev_list = df['trace_name'].to_list()

    # Preallocate the waveform array: (channels, samples, number_of_events)
    # Adjust sizes to match your data (here assumed 3 channels x 6000 samples x events)
    waveforms = np.zeros((3, 6000, num_samples), dtype=np.float32)

    # ObsPy FDSN client for removing response
    client = Client("IRIS")

    for i, trace_name in enumerate(ev_list):
        print(f"Processing {i + 1}/{len(ev_list)}: data/{trace_name}")

        # Retrieve the HDF5 dataset
        dataset = dtfl.get(f"data/{trace_name}")
        if dataset is None:
            print(f"Warning: Dataset {trace_name} not found in the HDF5 file.")
            continue

        # Convert the dataset to an ObsPy Stream
        st = make_stream(dataset)

        # Get station response metadata
        try:
            inventory = client.get_stations(
                network=dataset.attrs['network_code'],
                station=dataset.attrs['receiver_code'],
                starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                loc="*",
                channel="*",
                level="response"
            )

            # Remove instrument response to get acceleration (ACC)
            st.remove_response(inventory=inventory, output="ACC", plot=False)
        except Exception as e:
            print(f"Error retrieving or removing response: {e}")
            continue

        # Define trimming window: 5 seconds before P arrival to 60 seconds afterward
        starttime = (UTCDateTime(dataset.attrs['trace_start_time']) +
                     dataset.attrs['p_arrival_sample'] * 0.01 - 5)
        endtime = starttime + 60
        st.trim(starttime, endtime, pad=True, fill_value=0)

        # Combine the three channels into one NumPy array
        # st[0], st[1], st[2] correspond to N, E, Z in this example
        st_data = np.vstack([tr.data for tr in st])

        # Store the waveform data in the allocated array
        # Ensure st_data does not exceed your preallocated shape
        n_samples = min(st_data.shape[1], waveforms.shape[1])
        waveforms[:, :n_samples, i] = st_data[:, :n_samples]

    # Write all metadata and waveform data to the new HDF5 file
    with h5py.File(file_path, "w") as h5f:
        h5f.create_dataset("event_ID", data=event_ID)
        h5f.create_dataset("hypocentral_distance", data=hypocentral_distance)
        h5f.create_dataset("hypocentre_depth", data=hypocentre_depth)
        h5f.create_dataset("hypocentre_latitude", data=hypocentre_latitude)
        h5f.create_dataset("hypocentre_longitude", data=hypocentre_longitude)
        h5f.create_dataset("is_shallow_crustal", data=is_shallow_crustal)
        h5f.create_dataset("magnitude", data=magnitude)
        h5f.create_dataset("station_altitude", data=station_altitude)
        h5f.create_dataset("station_latitude", data=station_latitude)
        h5f.create_dataset("station_longitude", data=station_longitude)
        h5f.create_dataset("station_name", data=station_name.astype('S'))  # store as string
        h5f.create_dataset("station_network", data=station_network.astype('S'))  # store as string
        h5f.create_dataset("time_sample", data=time_sample)
        h5f.create_dataset("time_arrival", data=time_arrival)
        h5f.create_dataset("trace_start_time", data=trace_start_time.astype('S'))  # store as string
        h5f.create_dataset("vs30", data=vs30)
        h5f.create_dataset("waveforms", data=waveforms)

    print(f"HDF5 file '{file_path}' created successfully.")


def main():
    """
    Main function to demonstrate reading a CSV, filtering a DataFrame,
    and creating an HDF5 file.
    """
    file_name = "chunk2/chunk2.hdf5"
    csv_file = "chunk2/chunk2.csv"

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    print(f"Total events in CSV file: {len(df)}")

    # Open the HDF5 file containing the raw seismic data
    dtfl = h5py.File(file_name, 'r')

    # Filter the DataFrame for local earthquakes:
    # (1) category: 'earthquake_local'
    # (2) distance <= 200 km
    # (3) magnitude > 4
    df = df[
        (df.trace_category == 'earthquake_local') &
        (df.source_distance_km <= 200) &
        (df.source_magnitude > 4)
    ]
    print(f"Total events selected: {len(df)}")

    # Create a new HDF5 file with selected events
    output_file_path = "raw_waveforms.h5"
    create_h5_file(output_file_path, df, dtfl)

    # Close the original HDF5 file
    dtfl.close()


if __name__ == "__main__":
    main()

