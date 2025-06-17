#!/usr/bin/env python3
"""
This script processes quake‐wise waveform MAT files (archived by a separate script)
and performs record selection, preprocessing, and builds “gan” structures.
It is a Python “translation” of the MATLAB code by Men‐Andrin Meier (2022/07/29).
"""

import glob
import os
import time

import numpy as np
import scipy.io as sio
from obspy import Stream, read


# ---------------------------------------------------------------------------
# Helper functions to convert MATLAB structs (mat_struct) to dictionaries
# ---------------------------------------------------------------------------
def _check_keys(d):
    """
    Checks if entries in the dictionary are mat-objects. If yes,
    converts them to nested dictionaries.
    """
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = _tolist(d[key])
    return d


def _todict(matobj):
    """
    Recursively converts a mat_struct object to a dictionary.
    """
    d = {}
    for field in matobj._fieldnames:
        elem = getattr(matobj, field)
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[field] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[field] = _tolist(elem)
        else:
            d[field] = elem
    return d


def _tolist(ndarray):
    """
    Recursively converts an ndarray (which may contain mat_struct objects)
    to a list.
    """
    elem_list = []
    for elem in ndarray:
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(elem))
        elif isinstance(elem, np.ndarray):
            elem_list.append(_tolist(elem))
        else:
            elem_list.append(elem)
    return elem_list


# =============================================================================
# Helper functions (stubs or basic implementations)
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
        recs (dict): A dictionary containing record data. It is assumed that recs['z_fullnames']
                     is a list (or array) whose length determines the number of records.
        keepme (array-like): A boolean mask (list or NumPy array) indicating which records to keep.

    Returns:
        dict: The updated recs dictionary with filtered fields and an updated 'n' field.
    """
    # Ensure that keepme is a list of booleans for list filtering.
    if isinstance(keepme, np.ndarray):
        keepme = keepme.tolist()

    # Determine the expected number of records.
    n0 = len(recs["z_fullnames"])

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
    recs["n"] = len(recs["z_fullnames"])
    return recs


def parsave_eq(fname, eq):
    """
    Save the earthquake (eq) structure to a MATLAB .mat file.

    Parameters:
        fname (str): The output filename.
        eq (dict): The earthquake structure to save.
    """
    sio.savemat(fname, {"eq": eq})
    print(f"Saved eq structure to {outFile}")


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
projectName = "wfGAN_python"
archiveDir = "/Users/mameier/data/general/japan/bosai22/dl20220725/arx20220730"
# archiveDir  = '/Users/mameier/data/general/japan/bosai22/dl20220725/arx20220730_ltp_test'

print("\n" + "x" * 80 + "\n Create, print & check directory names \n")
if isBigstar:
    archiveDir = archiveDir.replace("/Users/mameier", "/scratch/kpalgunadi")

eqsFilesDir = os.path.join(archiveDir, "eqs")
projectDir = os.path.join(archiveDir, "proj", projectName)
outDir = os.path.join(archiveDir, "proj", projectName, "out")
figDir = os.path.join(archiveDir, "proj", projectName, "fig")
diaryFile = os.path.join(archiveDir, "proj", projectName, "proc_wform_archive.dairy")
stationFile = os.path.join(eqsFilesDir, "stationList.mat")

# Open a diary/log file (here simply for writing messages)
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

# Load station file (assumes a MATLAB .mat file)
station_data = sio.loadmat(stationFile)
station_data = _check_keys(station_data)

# Select MAT files via magnitude in file names ...........................
fList = glob.glob(os.path.join(eqsFilesDir, "M*km.mat"))
neq_tot = len(fList)

# Extract magnitude from file names.
# Assumes the file name is of the form e.g. 'M4p5km.mat'
selected = []
for fname in fList:
    base = os.path.basename(fname)
    mag_str = base[1:4].replace("p", ".")
    try:
        mag_val = float(mag_str)
    except ValueError:
        continue
    if p["mmin"] <= mag_val < p["mmax"]:
        selected.append((mag_val, fname))

# Sort selected files in descending order by magnitude
selected.sort(key=lambda x: x[0], reverse=True)
fList = [fname for mag, fname in selected]
neq = len(fList)

print("\n" + "x" * 80)
print(f"\n{projectName} project:")
print(f"\n{neq} / {neq_tot} found quakes meet selected magnitude criteria. Processing ... GO!")
print("\n" + "x" * 80 + "\n")
time.sleep(1)

# Loop over all earthquake events in the archive
for ieq, fname in enumerate(fList, start=1):
    print("\n-------------------------------------------------------------------------------")
    print(f"\nProcessing event {ieq}/{neq}: {fname}")

    # Load the earthquake structure from the MAT file
    tmp = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    tmp = _check_keys(tmp)
    eq = tmp["eq"]
    # (Depending on how the structure is loaded, you may need to convert it to a dict.)
    try:
        eq = eq.__dict__
    except AttributeError:
        pass  # already a dict

    # Extract recs (records) structure; again adjust if necessary.
    recs = eq["recs"]
    try:
        recs = recs.__dict__
    except AttributeError:
        pass

    # Decide if the event is shallow crustal
    try:
        if np.min(recs["rhyp"]) <= 60 and eq["dep"] <= 25:
            eq["is_shallow_crustal"] = True
        else:
            eq["is_shallow_crustal"] = False
    except:
        eq["is_shallow_crustal"] = []
        print(f"Number of recs is {len(recs['rhyp'])}")

    # Only keep records with hypocentral distance and depth within selected ranges
    zz = np.full(np.array(recs["rhyp"]).shape, eq["dep"], dtype=float)
    useme = (
        (np.array(recs["rhyp"]) >= p["rhmin"])
        & (np.array(recs["rhyp"]) <= p["rhmax"])
        & (zz >= p["zmin"])
        & (zz <= p["zmax"])
    )
    recs = select_record_subset(recs, useme)
    try:
        nrecs = recs.get("n", len(recs["rhyp"]))
    except:
        print("no record found")
        continue

    # If running on laptop, update full file paths in recs
    if isLaptop:
        recs["z_fullnames"] = [
            x.replace("/scratch/kpalgunadi", "/Users/mameier") for x in recs["z_fullnames"]
        ]
        recs["e_fullnames"] = [
            x.replace("/scratch/kpalgunadi", "/Users/mameier") for x in recs["e_fullnames"]
        ]
        recs["n_fullnames"] = [
            x.replace("/scratch/kpalgunadi", "/Users/mameier") for x in recs["n_fullnames"]
        ]

    # Print processing message (using recs['n0'] if available)
    title = eq["txt"]["title"]
    print(f"\nProcessing {nrecs}/{recs.get('n0', 'N/A')} waveforms of {title}")
    if nrecs < 1:
        print(f"Skipping {title}")
        continue

    # Allocate arrays for the three waveform components
    zmat = np.zeros((nrecs, nti))
    emat = np.zeros((nrecs, nti))
    nmat = np.zeros((nrecs, nti))

    # Loop over each record in this earthquake
    for irec in range(nrecs):
        # Read individual waveform components for Z, E, and N.
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

    # Remove skipped records (keep only where skipme==0)
    keepme = np.array(recs["skipme"]) == 0
    nkeep = int(np.sum(keepme))
    zmat = zmat[keepme, :]
    emat = emat[keepme, :]
    nmat = nmat[keepme, :]

    # Extract station information from recs.
    sta_id = np.array(recs["z_filenames"])[keepme]
    sta_network = [s[:2] for s in sta_id]  # first two characters
    sta_name = [s[2:6] for s in sta_id]  # characters 3 to 6
    sta_lat = np.array(recs["stLat"])[keepme]
    sta_lon = np.array(recs["stLon"])[keepme]
    sta_alt = np.array(recs["stAlt"])[keepme]

    vs30 = np.array(recs["vs30"])[keepme]
    vs30 = np.where(np.isnan(vs30), -1, vs30)

    # Merge 1c waveform matrices to a single 3c array (shape: [nkeep, nti, 3]).
    print("Merging 1c waveform matrices to single 3c matrix .. ", end="")
    wfMat = np.stack((nmat, emat, zmat))
    print("done.")

    # Retrieve focal mechanism parameters if available.
    try:
        strike = eq["sources"]["fm"]["strike"]
        dip = eq["sources"]["fm"]["dip"]
        rake = eq["sources"]["fm"]["rake"]
    except (KeyError, TypeError):
        strike = -999
        dip = -999
        rake = -999

    # Save upgraded earthquake structure under the 'gan' key.
    eq["gan"] = {}
    eq["gan"]["wfMat"] = wfMat
    eq["gan"]["componentOrder"] = "NEZ"
    # Convert origin time to string (datestr in MATLAB)
    eq["gan"]["t0"] = str(eq["t0"])
    eq["gan"]["vs30"] = vs30
    eq["gan"]["snr"] = np.array(recs["px_snr"])[keepme]
    eq["gan"]["rhyp"] = np.array(recs["rhyp"])[keepme]
    eq["gan"]["mag"] = np.full((nkeep,), eq["mag"])
    eq["gan"]["lat"] = np.full((nkeep,), eq["lat"])
    eq["gan"]["lon"] = np.full((nkeep,), eq["lon"])
    eq["gan"]["dep"] = np.full((nkeep,), eq["dep"])
    eq["gan"]["sta_network"] = sta_network
    eq["gan"]["sta_name"] = sta_name
    eq["gan"]["sta_lat"] = sta_lat
    eq["gan"]["sta_lon"] = sta_lon
    eq["gan"]["sta_alt"] = sta_alt
    eq["gan"]["is_shallow_crustal"] = np.array(np.full((nkeep,), eq["is_shallow_crustal"]))
    eq["gan"]["parameters"] = p
    eq["gan"]["strike"] = np.full((nkeep,), strike)
    eq["gan"]["dip"] = np.full((nkeep,), dip)
    eq["gan"]["rake"] = np.full((nkeep,), rake)

    # Save the processed earthquake if there is at least one record.
    if nkeep > 0:
        # Get the output file name from eq.files.eqFileName.
        # (This example assumes that eq['files'] is a dict with key 'eqFileName'.)
        try:
            eqFileName = eq["files"]["eqFileName"] + ".mat"
        except (KeyError, TypeError):
            eqFileName = f"eq_{ieq:03d}.mat"
        outFile = os.path.join(outDir, eqFileName)
        parsave_eq(outFile, eq)

print("\nProcessing completed.\n")
