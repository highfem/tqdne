#!/usr/bin/env python3
"""
This script processes earthquake waveform data from HDF5 files
and performs record selection, preprocessing, and builds "gan" structures.
Adapted to work with HDF5 files instead of MATLAB .mat files.
"""

import os
import glob
import time
import numpy as np
import h5py
from scipy.signal import butter, filtfilt
from obspy import Stream, read

# =============================================================================
# HDF5 reading functions
# =============================================================================


def read_earthquake_data_full(hdf5_path, file_index=0):
    """
    Read complete earthquake data structure from HDF5 file.

    Parameters:
    hdf5_path: Path to the HDF5 file
    file_index: Index of the file to read (0-based)

    Returns:
    Dictionary with the complete earthquake data structure
    """

    def read_group_recursive(group):
        """Recursively read HDF5 group into nested dictionary."""
        result = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                result[key] = read_group_recursive(group[key])
            else:
                data = group[key][()]
                if isinstance(data, bytes):
                    result[key] = data.decode()
                elif hasattr(data, "dtype") and data.dtype.kind in ["U", "S"]:
                    if data.size == 1:
                        result[key] = str(data.item())
                    else:
                        result[key] = [str(item) for item in data]
                else:
                    result[key] = data
        return result

    with h5py.File(hdf5_path, "r") as h5f:
        file_groups = [key for key in h5f.keys() if key.startswith("file_")]

        if file_index >= len(file_groups):
            raise IndexError(
                f"File index {file_index} not found. Available: 0-{len(file_groups) - 1}"
            )

        group_name = file_groups[file_index]
        group = h5f[group_name]

        # Read the complete earthquake structure
        eq_data = read_group_recursive(group)

        # Add metadata
        eq_data["_hdf5_group"] = group_name
        eq_data["_original_filename"] = group.attrs.get("original_filename", "unknown")

        return eq_data


def get_available_earthquakes(hdf5_path):
    """
    Get list of available earthquake records in HDF5 file.

    Returns:
    List of tuples: (index, group_name, original_filename, magnitude)
    """
    earthquakes = []
    with h5py.File(hdf5_path, "r") as h5f:
        file_groups = [key for key in h5f.keys() if key.startswith("file_")]

        for i, group_name in enumerate(file_groups):
            group = h5f[group_name]
            original_filename = group.attrs.get("original_filename", "unknown")

            # Try to get magnitude
            mag = -999
            if "mag" in group:
                mag = group["mag"][()]

            earthquakes.append((i, group_name, original_filename, mag))

    return earthquakes


# =============================================================================
# Helper functions
# =============================================================================


def print_iter_nums(i, total, step):
    """Print progress every 'step' iterations."""
    if i % step == 0:
        print(f"Processing record {i}/{total}")


def select_record_subset(recs, keepme):
    """
    Retain only the entries in each field of the recs dictionary for which the corresponding
    element in keepme is True. A field is filtered only if its length matches that of recs['z_fullnames'].

    Parameters:
        recs (dict): A dictionary containing record data.
        keepme (array-like): A boolean mask indicating which records to keep.

    Returns:
        dict: The updated recs dictionary with filtered fields and an updated 'n' field.
    """
    # Ensure that keepme is a list of booleans for list filtering.
    if isinstance(keepme, np.ndarray):
        keepme = keepme.tolist()

    # Determine the expected number of records.
    n0 = len(recs["z_fullnames"]) if "z_fullnames" in recs else 0

    # Loop over all fields in the dictionary.
    for key in list(recs.keys()):
        value = recs[key]
        try:
            # Check if the field's length matches that of z_fullnames.
            if hasattr(value, "__len__") and len(value) == n0:
                # If value is a NumPy array, use boolean indexing.
                if isinstance(value, np.ndarray):
                    recs[key] = value[keepme]
                else:
                    # Otherwise, assume value is a list and filter via list comprehension.
                    recs[key] = [v for v, keep in zip(value, keepme) if keep]
        except Exception:
            # If the field cannot be measured or indexed, leave it unchanged.
            pass

    # Update the number of records.
    if "z_fullnames" in recs:
        recs["n"] = len(recs["z_fullnames"])

    return recs


def write_dict_to_hdf5(group, data, name=None):
    """Write nested dictionary to HDF5 group."""
    if isinstance(data, dict):
        if name:
            subgroup = group.create_group(name)
        else:
            subgroup = group

        for key, value in data.items():
            write_dict_to_hdf5(subgroup, value, str(key))

    elif isinstance(data, (list, tuple)):
        try:
            arr = np.array(data)
            if name:
                group.create_dataset(name, data=arr)
        except (ValueError, TypeError):
            if name:
                dt = h5py.special_dtype(vlen=str)
                group.create_dataset(name, data=[str(item) for item in data], dtype=dt)

    elif isinstance(data, np.ndarray):
        if name:
            if data.dtype.kind in ["U", "S", "O"]:
                if data.size == 1:
                    group.create_dataset(name, data=str(data.item()))
                else:
                    dt = h5py.special_dtype(vlen=str)
                    string_data = [str(item) for item in data.flatten()]
                    group.create_dataset(name, data=string_data, dtype=dt)
            else:
                group.create_dataset(name, data=data)

    elif isinstance(data, (str, int, float, np.number)):
        if name:
            group.create_dataset(name, data=data)

    else:
        if name:
            group.create_dataset(name, data=str(data))


def save_all_processed_earthquakes(fname, processed_earthquakes, parameters):
    """
    Save all processed earthquake structures to a single HDF5 file.

    Parameters:
        fname (str): The output filename.
        processed_earthquakes (list): List of processed earthquake dictionaries.
        parameters (dict): Processing parameters.
    """
    with h5py.File(fname, "w") as h5f:
        # Add global metadata
        h5f.attrs["total_earthquakes"] = len(processed_earthquakes)
        h5f.attrs["project_name"] = "wfGAN_python_hdf5"
        h5f.attrs["processing_date"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save processing parameters
        param_group = h5f.create_group("processing_parameters")
        write_dict_to_hdf5(param_group, parameters)

        # Save each earthquake as a separate group
        for i, eq in enumerate(processed_earthquakes):
            eq_group_name = f"earthquake_{i:04d}"
            eq_group = h5f.create_group(eq_group_name)

            # Add earthquake metadata
            eq_group.attrs["earthquake_index"] = i
            if "_original_filename" in eq:
                eq_group.attrs["original_filename"] = eq["_original_filename"]
            if "mag" in eq:
                eq_group.attrs["magnitude"] = eq["mag"]

            # Write earthquake data (excluding internal metadata)
            eq_clean = {k: v for k, v in eq.items() if not k.startswith("_")}
            write_dict_to_hdf5(eq_group, eq_clean)

    print(f"Saved {len(processed_earthquakes)} processed earthquakes to {fname}")


# =============================================================================
# Main processing routine
# =============================================================================

# Determine system based on HOME environment variable
home = os.environ.get("HOME")
isLaptop = home == "/Users/mameier"
isBigstar = home == "/home/kpalgunadi"

print("\n" + "x" * 80)
print("\n Run start!!! \n")

# Parameters .............................................................
projectName = "wfGAN_python_hdf5"
archiveDir = "/Users/mameier/data/general/japan/bosai22/dl20220725/arx20220730"
hdf5_file = "earthquake_data.h5"  # HDF5 file containing converted earthquake data

print("\n" + "x" * 80 + "\n Create, print & check directory names \n")
if isBigstar:
    archiveDir = archiveDir.replace("/Users/mameier", "/scratch/kpalgunadi")

eqsFilesDir = os.path.join(archiveDir, "eqs")
projectDir = os.path.join(archiveDir, "proj", projectName)
outDir = os.path.join(archiveDir, "proj", projectName, "out")
figDir = os.path.join(archiveDir, "proj", projectName, "fig")
diaryFile = os.path.join(archiveDir, "proj", projectName, "proc_wform_archive.dairy")
hdf5_path = os.path.join(eqsFilesDir, hdf5_file)

# Create output directories if they don't exist
os.makedirs(outDir, exist_ok=True)
os.makedirs(figDir, exist_ok=True)
os.makedirs(projectDir, exist_ok=True)

# Open a diary/log file
with open(diaryFile, "w") as diary:
    diary.write("Processing started...\n")

print("\n" + "x" * 80 + "\n Record selection process.... \n")

# Record selection parameters ..............................................
p = {}
p["rhmin"] = 0
p["rhmax"] = 200
p["mmin"] = 4.0
p["mmax"] = 10.0
p["zmin"] = -5
p["zmax"] = 100

# Preprocessing parameters
p["preproc"] = {"nstap": 100, "forder": 2, "type": "causal", "flo": 0.1}

# GAN parameters: define a common time vector from -5 to 120 seconds
p["gan"] = {}
p["gan"]["ti"] = np.arange(-5, 120 + 0.01, 0.01)
nti = len(p["gan"]["ti"])

# Get available earthquakes from HDF5 file
available_earthquakes = get_available_earthquakes(hdf5_path)
neq_tot = len(available_earthquakes)

# Filter earthquakes by magnitude
selected = []
for idx, group_name, filename, mag in available_earthquakes:
    if p["mmin"] <= mag < p["mmax"]:
        selected.append((mag, idx, group_name, filename))

# Sort selected earthquakes in descending order by magnitude
selected.sort(key=lambda x: x[0], reverse=True)
neq = len(selected)

print("\n" + "x" * 80)
print(f"\n{projectName} project:")
print(f"\n{neq} / {neq_tot} found quakes meet selected magnitude criteria. Processing ... GO!")
print("\n" + "x" * 80 + "\n")
time.sleep(1)

# Initialize list to store all processed earthquakes
processed_earthquakes = []

# Loop over all selected earthquake events
for ieq, (mag, file_index, group_name, filename) in enumerate(selected, start=1):
    print(f"\n-------------------------------------------------------------------------------")
    print(f"\nProcessing event {ieq}/{neq}: {filename} (magnitude: {mag:.1f})")

    # Load the earthquake structure from HDF5 file
    eq = read_earthquake_data_full(hdf5_path, file_index)

    # Extract recs (records) structure
    recs = eq.get("recs", {})
    if not recs:
        print(f"No records found for {filename}")
        continue

    # Decide if the event is shallow crustal
    try:
        rhyp_array = np.array(recs["rhyp"]) if "rhyp" in recs else np.array([])
        if len(rhyp_array) > 0 and np.min(rhyp_array) <= 60 and eq.get("dep", 0) <= 25:
            eq["is_shallow_crustal"] = True
        else:
            eq["is_shallow_crustal"] = False
    except:
        eq["is_shallow_crustal"] = False
        print(f"Could not determine shallow crustal status")

    # Only keep records with hypocentral distance and depth within selected ranges
    if "rhyp" not in recs or len(recs["rhyp"]) == 0:
        print(f"No rhyp data found for {filename}")
        continue

    rhyp_array = np.array(recs["rhyp"])
    zz = np.full(rhyp_array.shape, eq.get("dep", 0), dtype=float)
    useme = (
        (rhyp_array >= p["rhmin"])
        & (rhyp_array <= p["rhmax"])
        & (zz >= p["zmin"])
        & (zz <= p["zmax"])
    )

    recs = select_record_subset(recs, useme)
    nrecs = recs.get("n", 0)

    if nrecs < 1:
        print(f"No records meet criteria for {filename}")
        continue

    # Print processing message
    title = eq.get("txt", {}).get("title", filename) if "txt" in eq else filename
    print(f"\nProcessing {nrecs}/{recs.get('n0', 'N/A')} waveforms of {title}")

    # Allocate arrays for the three waveform components
    zmat = np.zeros((nrecs, nti))
    emat = np.zeros((nrecs, nti))
    nmat = np.zeros((nrecs, nti))

    # Loop over each record in this earthquake
    for irec in range(nrecs):
        try:
            # Read individual waveform components for Z, E, and N
            st = Stream()
            z = read(recs["z_fullnames"][irec], format="KNET", apply_calib=True)
            e = read(recs["e_fullnames"][irec], format="KNET", apply_calib=True)
            n_comp = read(recs["n_fullnames"][irec], format="KNET", apply_calib=True)
            st = z + e + n_comp

            # Pre-process each component (apply highpass filtering)
            st.detrend("demean")
            st.detrend("linear")
            st.filter("highpass", freq=p["preproc"]["flo"], zerophase=False)

            len_data = min(len(st[0].data), nti)

            zmat[irec, :len_data] = st[0].data[:len_data]
            emat[irec, :len_data] = st[1].data[:len_data]
            nmat[irec, :len_data] = st[2].data[:len_data]

        except Exception as e:
            print(f"Error processing record {irec}: {e}")
            continue

    # Remove skipped records (keep only where skipme==0)
    if "skipme" in recs:
        keepme = np.array(recs["skipme"]) == 0
    else:
        keepme = np.ones(nrecs, dtype=bool)

    nkeep = int(np.sum(keepme))
    if nkeep == 0:
        print(f"All records skipped for {filename}")
        continue

    zmat = zmat[keepme, :]
    emat = emat[keepme, :]
    nmat = nmat[keepme, :]

    # Extract station information from recs
    def safe_array_extract(data, keepme, default_val=""):
        """Safely extract and filter array data."""
        if data is None:
            return [default_val] * nkeep
        try:
            arr = np.array(data)
            if len(arr) == len(keepme):
                return arr[keepme]
            else:
                return [default_val] * nkeep
        except:
            return [default_val] * nkeep

    sta_id = safe_array_extract(recs.get("z_filenames"), keepme, "UNKN")
    sta_network = [str(s)[:2] if len(str(s)) >= 2 else "UN" for s in sta_id]
    sta_name = [str(s)[2:6] if len(str(s)) >= 6 else str(s) for s in sta_id]

    sta_lat = safe_array_extract(recs.get("stLat"), keepme, -999)
    sta_lon = safe_array_extract(recs.get("stLon"), keepme, -999)
    sta_alt = safe_array_extract(recs.get("stAlt"), keepme, -999)

    vs30 = safe_array_extract(recs.get("vs30"), keepme, -1)
    vs30 = np.where(np.isnan(vs30), -1, vs30)

    # Merge 1c waveform matrices to a single 3c array (shape: [3, nkeep, nti])
    print("Merging 1c waveform matrices to single 3c matrix .. ", end="")
    wfMat = np.stack((nmat, emat, zmat))
    print("done.")

    # Retrieve focal mechanism parameters if available
    try:
        sources = eq.get("sources", {})
        fm = sources.get("fm", {}) if sources else {}
        strike = fm.get("strike", -999)
        dip = fm.get("dip", -999)
        rake = fm.get("rake", -999)
    except:
        strike = -999
        dip = -999
        rake = -999

    # Save upgraded earthquake structure under the 'gan' key
    eq["gan"] = {}
    eq["gan"]["wfMat"] = wfMat
    eq["gan"]["componentOrder"] = "NEZ"
    eq["gan"]["t0"] = str(eq.get("t0", ""))
    eq["gan"]["vs30"] = vs30
    eq["gan"]["snr"] = safe_array_extract(recs.get("px_snr"), keepme, -999)
    eq["gan"]["rhyp"] = safe_array_extract(recs.get("rhyp"), keepme, -999)
    eq["gan"]["mag"] = np.full((nkeep,), eq.get("mag", -999))
    eq["gan"]["lat"] = np.full((nkeep,), eq.get("lat", -999))
    eq["gan"]["lon"] = np.full((nkeep,), eq.get("lon", -999))
    eq["gan"]["dep"] = np.full((nkeep,), eq.get("dep", -999))
    eq["gan"]["sta_network"] = sta_network
    eq["gan"]["sta_name"] = sta_name
    eq["gan"]["sta_lat"] = sta_lat
    eq["gan"]["sta_lon"] = sta_lon
    eq["gan"]["sta_alt"] = sta_alt
    eq["gan"]["is_shallow_crustal"] = np.full((nkeep,), eq["is_shallow_crustal"])
    eq["gan"]["parameters"] = p
    eq["gan"]["strike"] = np.full((nkeep,), strike)
    eq["gan"]["dip"] = np.full((nkeep,), dip)
    eq["gan"]["rake"] = np.full((nkeep,), rake)

    # Add processed earthquake to list if there is at least one record
    if nkeep > 0:
        # Add processing metadata
        eq["_processing_index"] = ieq
        eq["_records_kept"] = nkeep
        eq["_records_total"] = recs.get("n0", nrecs)

        processed_earthquakes.append(eq)
        print(f"Added earthquake {ieq} to processing list ({nkeep} records)")

# Save all processed earthquakes to a single HDF5 file
if processed_earthquakes:
    final_output_file = os.path.join(outDir, f"{projectName}_processed_earthquakes.h5")
    save_all_processed_earthquakes(final_output_file, processed_earthquakes, p)

    print(f"\n" + "x" * 80)
    print(f"\nSUCCESS: Processed {len(processed_earthquakes)} earthquakes")
    print(f"Final output saved to: {final_output_file}")
    print(
        f"Total records across all earthquakes: {sum(eq['_records_kept'] for eq in processed_earthquakes)}"
    )
    print("\n" + "x" * 80)
else:
    print(f"\nWARNING: No earthquakes were successfully processed!")

print("\nProcessing completed.\n")
