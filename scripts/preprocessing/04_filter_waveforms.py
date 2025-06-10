import h5py
import numpy as np
import matplotlib.pyplot as plt
import seisbench.models as sbm
from obspy import Stream, Trace, UTCDateTime

h5_file_path = "raw_waveforms.h5"

# Dictionary to hold each variable as a NumPy array
data_dict = {}

# Open the HDF5 file in read-only mode
with h5py.File(h5_file_path, 'r') as file:
    # List all available keys (datasets)
    keys = list(file.keys())
    print("Available keys:", keys)
    
    # Loop over the provided keys and save each dataset to a dictionary as a NumPy array
    for key in keys:
        data_dict[key] = np.array(file[key])
        print(f"Loaded '{key}' with shape {data_dict[key].shape}")

azimuth = data_dict['azimuth']
azimuthal_gap = data_dict['azimuthal_gap']
back_azimuth = data_dict['back_azimuth']
event_ID = data_dict['event_ID']
hypocentral_distance = data_dict['hypocentral_distance']
hypocentre_depth = data_dict['hypocentre_depth']
hypocentre_dip = data_dict['hypocentre_dip']
hypocentre_latitude = data_dict['hypocentre_latitude']
hypocentre_longitude = data_dict['hypocentre_longitude']
hypocentre_rake = data_dict['hypocentre_rake']
hypocentre_strike = data_dict['hypocentre_strike']
is_onshore = data_dict['is_onshore']
magnitude = data_dict['magnitude']
station_latitude = data_dict['station_latitude']
station_longitude = data_dict['station_longitude']
station_name = data_dict['station_name']
station_network = data_dict['station_network']
trace_sampling_rate = data_dict['trace_sampling_rate']
waveforms = data_dict['waveforms']
vs30 = data_dict['vs30']
z_filename = data_dict['z_filename']

idx_vs30 = np.where(vs30 <= 0)[0]
station_means = waveforms.max(axis=(1, 2))
tol = 1e-8
mask = np.isclose(station_means, 0.0, atol=tol)
station_indices = np.where(mask)[0]
remove_station_idx = np.unique(np.hstack((station_indices, idx_vs30)))

waveforms = np.delete(waveforms, remove_station_idx, axis=0)
azimuth = np.delete(azimuth, remove_station_idx)
azimuthal_gap = np.delete(azimuthal_gap, remove_station_idx)
back_azimuth = np.delete(back_azimuth, remove_station_idx)
event_ID = np.delete(event_ID, remove_station_idx)
hypocentral_distance = np.delete(hypocentral_distance, remove_station_idx)
hypocentre_depth = np.delete(hypocentre_depth, remove_station_idx)
hypocentre_dip = np.delete(hypocentre_dip, remove_station_idx)
hypocentre_latitude = np.delete(hypocentre_latitude, remove_station_idx)
hypocentre_longitude = np.delete(hypocentre_longitude, remove_station_idx)
hypocentre_rake = np.delete(hypocentre_rake, remove_station_idx)
hypocentre_strike = np.delete(hypocentre_strike, remove_station_idx)
is_onshore = np.delete(is_onshore, remove_station_idx)
magnitude = np.delete(magnitude, remove_station_idx)
station_latitude = np.delete(station_latitude, remove_station_idx)
station_longitude = np.delete(station_longitude, remove_station_idx)
station_name = np.delete(station_name, remove_station_idx)
station_network = np.delete(station_network, remove_station_idx)
trace_sampling_rate = np.delete(trace_sampling_rate, remove_station_idx)
vs30 = np.delete(vs30, remove_station_idx)
z_filename = np.delete(z_filename, remove_station_idx)

## recheck the waveform

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

    # integrate to velocity
    tr.integrate()  
    return tr

# Load models
model = sbm.PhaseNet.from_pretrained("jma")
failed_waveform_index = []
for idx in range(len(vs30)):
    st = Stream()
    # Shared trace stats
    stats_dict = {
        'sampling_rate': 1 / 0.01,
        'delta': 0.01,
        'starttime': UTCDateTime("2025-02-19T05:00:00"),
        'network': station_network[idx],
        'station': station_name[idx],
    }
    
    # Channels for NEZ and their respective slices in data_array
    channel_map = {
        'HHN': waveforms[idx,:5500,0],
        'HHE': waveforms[idx,:5500,1],
        'HHZ': waveforms[idx,:5500,2],
    }

    for channel, waveform in channel_map.items():
        st += create_trace(waveform, stats_dict.copy(), channel)


    # Classify with PhaseNet, grouping traces in triplets (N, E, Z)
    pn_predictor = []
    pn_P = []
    pn_S = []
    pn_peak_time_P = []
    pn_peak_time_S = []
    pn_trace_id = []
    pn_peak_value_P = []
    pn_peak_value_S = []
    
    st_3c = st 
    st_3c.detrend("demean")
    st_3c.detrend("linear")
    st_3c.filter("bandpass", freqmin=0.1, freqmax=30, zerophase=False)

    # Record the predictor name
    pn_predictor.append('PhaseNet-jma')
    annotations = model.annotate(st_3c)

    # Run classification
    output = model.classify(st_3c, P_threshold=0.05, S_threshold=0.04)
    pn_picks = output.picks

    if pn_picks:
        # Grab trace ID from the first pick for labeling
        first_pick_dict = pn_picks[0].__dict__
        pn_trace_id.append(first_pick_dict['trace_id'])
        
        # Variables to check if we have appended a P and an S
        p_has_been_appended = False
        s_has_been_appended = False
        
        for pick in pn_picks:
            pick_info = pick.__dict__
            phase = pick_info['phase']
            
            # If we find a P pick for the first time
            if phase == 'P' and not p_has_been_appended:
                pn_P.append('P')
                pn_peak_time_P.append(pick_info['peak_time'])
                pn_peak_value_P.append(pick_info['peak_value'])
                p_has_been_appended = True
            
            # If we find an S pick for the first time
            elif phase == 'S' and not s_has_been_appended:
                pn_S.append('S')
                pn_peak_time_S.append(pick_info['peak_time'])
                pn_peak_value_S.append(pick_info['peak_value'])
                s_has_been_appended = True
            
            
            # Handle the scenario where there is only one pick total:
            # If that pick is P only, append default S or vice versa.
            if phase == 'P' and len(pn_picks) == 1:
                pn_S.append('S')
                pn_peak_time_S.append(np.nan)
                pn_peak_value_S.append(np.nan)
                s_has_been_appended = True
            elif phase == 'S' and len(pn_picks) == 1:
                pn_P.append('P')
                pn_peak_time_P.append(np.nan)
                pn_peak_value_P.append(np.nan)
                p_has_been_appended = True
        
        # If we never found a P pick
        if not p_has_been_appended:
            pn_P.append('P')
            pn_peak_time_P.append(np.nan)
            pn_peak_value_P.append(np.nan)
        
        # If we never found an S pick
        if not s_has_been_appended:
            pn_S.append('S')
            pn_peak_time_S.append(np.nan)
            pn_peak_value_S.append(np.nan)
    
    else:
        # No picks found at all
        default_trace_id = (
            f"{st_3c[0].stats.network}."
            f"{st_3c[0].stats.station}."
        )
        pn_trace_id.append(default_trace_id)
        
        pn_P.append('P')
        pn_peak_time_P.append(np.nan)
        pn_peak_value_P.append(np.nan)
        
        pn_S.append('S')
        pn_peak_time_S.append(np.nan)
        pn_peak_value_S.append(np.nan)

    try:
        time_diff = np.round(pn_peak_time_P[0] - st_3c[0].stats.starttime, decimals=1)
        if time_diff > 7.0 or time_diff < 2.0:
            print(f"Index {idx}: time = {pn_peak_time_P[0] - st_3c[0].stats.starttime}")
            failed_waveform_index.append(idx)
            #if np.mod(idx, 1000):
            #    st_3c.plot(outfile=f'figures/{idx}_{time_diff}.png')
    except:
        print(f"Index {idx}: failed waveforms")
        failed_waveform_index.append(idx)
        #if np.mod(idx, 1000):
        #    st_3c.plot(outfile=f'figures/{idx}_failed_waveform.png')

failed_waveform_index = np.asarray(failed_waveform_index)
print(f"\n The total filtered bad waveforms is {len(failed_waveform_index)}")
print(f"The left over number of data is {len(azimuth) - len(failed_waveform_index)}")

waveforms = np.delete(waveforms, failed_waveform_index, axis=0)
azimuth = np.delete(azimuth, failed_waveform_index)
azimuthal_gap = np.delete(azimuthal_gap, failed_waveform_index)
back_azimuth = np.delete(back_azimuth, failed_waveform_index)
event_ID = np.delete(event_ID, failed_waveform_index)
hypocentral_distance = np.delete(hypocentral_distance, failed_waveform_index)
hypocentre_depth = np.delete(hypocentre_depth, failed_waveform_index)
hypocentre_dip = np.delete(hypocentre_dip, failed_waveform_index)
hypocentre_latitude = np.delete(hypocentre_latitude, failed_waveform_index)
hypocentre_longitude = np.delete(hypocentre_longitude, failed_waveform_index)
hypocentre_rake = np.delete(hypocentre_rake, failed_waveform_index)
hypocentre_strike = np.delete(hypocentre_strike, failed_waveform_index)
is_onshore = np.delete(is_onshore, failed_waveform_index)
magnitude = np.delete(magnitude, failed_waveform_index)
station_latitude = np.delete(station_latitude, failed_waveform_index)
station_longitude = np.delete(station_longitude, failed_waveform_index)
station_name = np.delete(station_name, failed_waveform_index)
station_network = np.delete(station_network, failed_waveform_index)
trace_sampling_rate = np.delete(trace_sampling_rate, failed_waveform_index)
vs30 = np.delete(vs30, failed_waveform_index)
z_filename = np.delete(z_filename, failed_waveform_index)

## Save in h5 format

file_path = "raw_waveforms_filtered.h5"
with h5py.File(file_path, "w") as h5f:
    h5f.create_dataset("event_ID", data=event_ID)
    h5f.create_dataset("hypocentral_distance", data=hypocentral_distance*1e-3)
    h5f.create_dataset("hypocentre_depth", data=hypocentre_depth )
    h5f.create_dataset("hypocentre_latitude", data=hypocentre_latitude )
    h5f.create_dataset("hypocentre_longitude", data=hypocentre_longitude )
    h5f.create_dataset("is_onshore", data=is_onshore )
    h5f.create_dataset("magnitude", data=magnitude )
    h5f.create_dataset("station_latitude", data=station_latitude )
    h5f.create_dataset("station_longitude", data=station_longitude )
    h5f.create_dataset("station_name", data=station_name )
    h5f.create_dataset("station_network", data=station_network )
    h5f.create_dataset("azimuth", data=azimuth )
    h5f.create_dataset("back_azimuth", data=back_azimuth )
    h5f.create_dataset("azimuthal_gap", data=azimuthal_gap )
    h5f.create_dataset("hypocentre_strike", data=hypocentre_strike )
    h5f.create_dataset("hypocentre_dip", data=hypocentre_dip )
    h5f.create_dataset("hypocentre_rake", data=hypocentre_rake )
    h5f.create_dataset("trace_sampling_rate", data=trace_sampling_rate )
    h5f.create_dataset("vs30", data=vs30 )
    h5f.create_dataset("z_filename", data=z_filename )
    h5f.create_dataset("waveforms", data=waveforms )

print(f"\nHDF5 file '{file_path}' created successfully.")
