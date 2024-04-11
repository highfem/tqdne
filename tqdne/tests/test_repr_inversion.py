import inspect
import sys
import h5py
from tqdne.representations import *

def test_repr_inversion(num_samples=20, datapath=Config().datasetdir / Config().data_test):
    # Get all the representation classes in tqdne.representations that are subclasses of Representation
    reprs = [m[1] for m in inspect.getmembers(sys.modules['tqdne.representations'], inspect.isclass)
               if issubclass(m[1], Representation) and m[1] != Representation]
    
    with h5py.File(datapath, "r", locking=False) as f: 
        waveforms = f["waveform"][:num_samples, :, :Config().signal_length]
    
    for repr in reprs:
        repr.test(repr, waveforms)

if __name__ == "__main__":
    test_repr_inversion()