"""
Script Name: create h5 for tqdne using STEAD dataset
Description:
    This script reads a CSV file and an HDF5 file containing seismic trace information. It writes relevant information into a new HDF5 file
    for tqdne structure.

"""

import h5py
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.geodetics.base import gps2dist_azimuth


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
    tr_E.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_E.stats.delta = 0.01  # Sample rate interval
    tr_E.stats.channel = dataset.attrs["receiver_type"] + "E"
    tr_E.stats.station = dataset.attrs["receiver_code"]
    tr_E.stats.network = dataset.attrs["network_code"]

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs["receiver_type"] + "N"
    tr_N.stats.station = dataset.attrs["receiver_code"]
    tr_N.stats.network = dataset.attrs["network_code"]

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs["receiver_type"] + "Z"
    tr_Z.stats.station = dataset.attrs["receiver_code"]
    tr_Z.stats.network = dataset.attrs["network_code"]

    # Create a Stream containing the three traces (N, E, Z)
    stream = obspy.Stream([tr_N, tr_E, tr_Z])

    return stream
    
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
        return azimuth
    
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


def create_h5_file(file_path, df, dtfl):
    """
    Creates an HDF5 file in a tqdne structure
    """

    event_ID_list = []
    hypocentral_distance_list = []
    hypocentre_depth_list = []
    hypocentre_latitude_list = []
    hypocentre_longitude_list = []
    is_shallow_crustal_list = []
    magnitude_list = []
    station_altitude_list = []
    station_latitude_list = []
    station_longitude_list = []
    station_name_list = []
    station_network_list = []
    time_sample_list = []
    time_arrival_list = []
    trace_start_time_list = []
    vs30_list = []
    waveform_list = []  # will hold arrays of shape (n_samples, 3) for each event
    azimuthal_gap = []

    # ObsPy FDSN client for removing response
    client = Client("IRIS")
    iter = 0
    for i, row in df.iterrows():
        iter += 1
        trace_name = row["trace_name"]
        dataset = dtfl.get(f"data/{trace_name}")

        if dataset is None:
            print(f"Warning: Dataset {trace_name} not found in the HDF5 file. Skipping...")
            continue

        try:
            inventory = client.get_stations(
                network=dataset.attrs["network_code"],
                station=dataset.attrs["receiver_code"],
                starttime=UTCDateTime(dataset.attrs["trace_start_time"]),
                endtime=UTCDateTime(dataset.attrs["trace_start_time"]) + 60,
                loc="*",
                channel="*",
                level="response",
            )
        except Exception as e:
            print(f"Error retrieving station metadata for {trace_name}: {e}")
            # Skip this event if station metadata is not available
            continue

        st = make_stream(dataset)
        try:
            st.remove_response(inventory=inventory, output="ACC", plot=False)
        except Exception as e:
            print(f"Error removing instrument response for {trace_name}: {e}")
            continue

        # Trim around the P arrival (5 seconds before, up to 60 seconds total)
        starttime = (
            UTCDateTime(dataset.attrs["trace_start_time"])
            + dataset.attrs["p_arrival_sample"] * 0.01
            - 5
        )
        endtime = starttime + 60
        st.trim(starttime, endtime, pad=True, fill_value=0)

        try:
            st_data = np.vstack([tr.data for tr in st])
        except Exception as e:
            print(f"Error stacking channels for {trace_name}: {e}")
            continue

        # For demonstration, we assume 6000 samples max
        max_samples = 6000
        # slice if st_data is longer than 6000
        st_data_clipped = st_data[:, :max_samples].T

        # store waveforms
        waveform_list.append(st_data_clipped)

        # store metadata
        event_ID_list.append(row["source_id"])
        hypocentral_distance_list.append(row["source_distance_km"])
        hypocentre_depth_list.append(row["source_depth_km"])
        hypocentre_latitude_list.append(row["source_latitude"])
        hypocentre_longitude_list.append(row["source_longitude"])
        is_shallow_crustal_list.append(1 if row["source_depth_km"] <= 25 else 0)
        magnitude_list.append(row["source_magnitude"])
        station_altitude_list.append(row["receiver_elevation_m"])
        station_latitude_list.append(row["receiver_latitude"])
        station_longitude_list.append(row["receiver_longitude"])
        station_name_list.append(row["receiver_code"])
        station_network_list.append(row["network_code"])
        time_sample_list.append(row["p_arrival_sample"])
        time_arrival_list.append(row["p_travel_sec"])
        trace_start_time_list.append(row["trace_start_time"])
        # vs30 could be real data or random for demonstration
        vs30_list.append(np.random.randint(400, 1501))
        
        # azimuthal gap calculation
        stations = np.vstack((row["receiver_latitude"], row["receiver_longitude"])).T
        hypo = (row["source_latitude"], row["source_longitude"])
        azimuthal_gap.append(calculate_azimuthal_gap(hypo, stations))

        print(f"Processed {iter}/{len(df)}: {trace_name} successfully.")

    event_ID_arr = np.array(event_ID_list)
    hypocentral_distance_arr = np.array(hypocentral_distance_list)
    hypocentre_depth_arr = np.array(hypocentre_depth_list)
    hypocentre_latitude_arr = np.array(hypocentre_latitude_list)
    hypocentre_longitude_arr = np.array(hypocentre_longitude_list)
    is_shallow_crustal_arr = np.array(is_shallow_crustal_list)
    magnitude_arr = np.array(magnitude_list)
    station_altitude_arr = np.array(station_altitude_list)
    station_latitude_arr = np.array(station_latitude_list)
    station_longitude_arr = np.array(station_longitude_list)
    station_name_arr = np.array(station_name_list).astype("S")
    station_network_arr = np.array(station_network_list).astype("S")
    time_sample_arr = np.array(time_sample_list)
    time_arrival_arr = np.array(time_arrival_list)
    trace_start_time_arr = np.array(trace_start_time_list).astype("S")
    vs30_arr = np.array(vs30_list)
    azimuthal_gap_arr = np.array(azimuthal_gap)

    # Waveforms: we need shape (n_events, 6000, 3)
    # waveform_list is a list of (3, <=6000) arrays
    n_events = len(waveform_list)
    waveforms_arr = np.zeros((n_events, max_samples, 3), dtype=np.float32)
    for i in range(n_events):
        n_samps = waveform_list[i].shape[1]
        waveforms_arr[i, :n_samps, :] = waveform_list[i]

    with h5py.File(file_path, "w") as h5f:
        h5f.create_dataset("event_ID", data=event_ID_arr)
        h5f.create_dataset("hypocentral_distance", data=hypocentral_distance_arr)
        h5f.create_dataset("hypocentre_depth", data=hypocentre_depth_arr)
        h5f.create_dataset("hypocentre_latitude", data=hypocentre_latitude_arr)
        h5f.create_dataset("hypocentre_longitude", data=hypocentre_longitude_arr)
        h5f.create_dataset("is_shallow_crustal", data=is_shallow_crustal_arr)
        h5f.create_dataset("magnitude", data=magnitude_arr)
        h5f.create_dataset("station_altitude", data=station_altitude_arr)
        h5f.create_dataset("station_latitude", data=station_latitude_arr)
        h5f.create_dataset("station_longitude", data=station_longitude_arr)
        h5f.create_dataset("station_name", data=station_name_arr)
        h5f.create_dataset("station_network", data=station_network_arr)
        h5f.create_dataset("time_sample", data=time_sample_arr)
        h5f.create_dataset("time_arrival", data=time_arrival_arr)
        h5f.create_dataset("trace_start_time", data=trace_start_time_arr)
        h5f.create_dataset("vs30", data=vs30_arr)
        h5f.create_dataset("waveforms", data=waveforms_arr)
        h5f.create_dataset("azimuthal_gap", data=azimuthal_gap_arr)

    print(f"\nHDF5 file '{file_path}' created successfully!")


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
    dtfl = h5py.File(file_name, "r")

    # Filter the DataFrame for local earthquakes:
    # (1) category: 'earthquake_local'
    # (2) distance <= 200 km
    # (3) magnitude > 4
    df = df[
        (df.trace_category == "earthquake_local")
        & (df.source_distance_km <= 200)
        & (df.source_magnitude > 4)
    ]
    print(f"Total events selected: {len(df)}")

    # Create a new HDF5 file with selected events
    output_file_path = "raw_waveforms.h5"
    create_h5_file(output_file_path, df, dtfl)

    # Close the original HDF5 file
    dtfl.close()


if __name__ == "__main__":
    main()
