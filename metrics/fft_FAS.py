import numpy as np 
import matplotlib.pyplot as plt
from example_GMM import calculate_gmfs
import scipy.integrate as it
from scipy.signal import resample
from utils import MatFileHandler, calculate_gmrotd50
import pickle
plt.rcParams.update({'font.size': 16})
from tqdne.conf import Config
from tqdne.dataset import SampleDataset
from tqdne.representations import Signal


config = Config()
dataset_path = "/store/sdsc/sd28/wforms_GAN_input_v20220805.h5"
train_dataset = SampleDataset(config.datasetdir / config.data_train, Signal())
test_dataset = SampleDataset(config.datasetdir / config.data_test, Signal())

#combine train and test datasets
waveforms = np.concatenate([train_dataset.waveforms, test_dataset.waveforms])
features = np.concatenate([train_dataset.features, test_dataset.features])

# define distance bins
dist_bins = np.array([0, 30, 50, 70, 90, 110, 150, 170])
mag_bins = np.array([4.5, 5, 5.5, 6.0, 6.5, 7, 7.5, 8])
imag = 3
idist = [0]

# divide samples in dist_bins
dist_bins_samples = []
waveforms_samples = []

max_num = 1000
for i in idist:
    mask1 = (features[:, 0] >= dist_bins[i]) & (features[:, 0] < dist_bins[i + 1])
    mask2 = (features[:, 2] >= mag_bins[imag]) & (features[:, 2] < mag_bins[imag + 1])
    mask3 = (features[:, 3] > 0)
    mask = mask1 & mask2 & mask3
    waveforms_samples.append(np.nan_to_num((waveforms[mask][:max_num])))
    dist_bins_samples.append(np.nan_to_num((features[mask,0][:max_num])))
# magnitude
mags = np.round((mag_bins[imag]+mag_bins[imag+1])/2, decimals=1)

dataset_folder = 'GM0_waveforms/'+str(mags)+'/'
with open(dataset_folder + 'shakemap_gm0-stft_'+str(mags)+'-'+str(dist_bins[idist[0]])+'-'+str(dist_bins[idist[0]+1])+'.pkl', 'rb') as f:
    shakemap_data_gen = pickle.load(f)
    print(shakemap_data_gen['waveforms'].shape, shakemap_data_gen['cond'].shape)

conditional = shakemap_data_gen['cond']
wf = shakemap_data_gen['waveforms']

rhyp = conditional[:,0]
is_shallow_crustal = conditional[:,1]
magnitude = conditional[:,2]
vs30 = conditional[:,3]

dt = 0.01
g_inv = 1/9.80665
st_tqdne_EW = wf[:,1,:]
st_tqdne_NS = wf[:,0,:]

wf_EW = waveforms_samples[0][:,1,:]
wf_NS = waveforms_samples[0][:,0,:]

print(st_tqdne_EW.shape, wf_EW.shape)
print(dist_bins[idist[0]], dist_bins[idist[0]+1])
### PGA
import pyrotd
import obspy as ob
from obspy.core.inventory.inventory import read_inventory

fft_gm0 = []
for i in range(len(st_tqdne_EW[:,0])):
    print(i, ' / ',len(st_tqdne_EW[:,0]))
    EW_gm0 = st_tqdne_EW[i,:]
    NS_gm0 = st_tqdne_NS[i,:]
    Ah = np.sqrt(0.5*( EW_gm0**2 + NS_gm0**2 ))
    # Compute FFT
    n = len(Ah)
    fft_values = np.fft.fft(Ah)
    fft_freq = np.fft.fftfreq(n, d=dt)
    
    # Get the positive frequencies only
    positive_freq_indices = np.where(fft_freq >= 0)
    fft_values_gm0 = fft_values[positive_freq_indices]
    fft_freq_gm0 = fft_freq[positive_freq_indices]
    fft_gm0.append(np.abs(fft_values_gm0))

fft_gm0 = np.array(fft_gm0)

fft_obs = []
for i in range(len(wf_EW[:,0])):
    print(i,' / ',len(wf_EW[:,0]))
    EW_obs = wf_EW[i,:]
    NS_obs = wf_NS[i,:]
    Ah_obs = np.sqrt(0.5*( EW_obs**2 + NS_obs**2 ))
    # Compute FFT
    n = len(Ah_obs)
    fft_values = np.fft.fft(Ah_obs)
    fft_freq = np.fft.fftfreq(n, d=dt)

    # Get the positive frequencies only
    positive_freq_indices = np.where(fft_freq >= 0)
    fft_values_obs = fft_values[positive_freq_indices]
    fft_freq_obs = fft_freq[positive_freq_indices]
    fft_obs.append(np.abs(fft_values_obs))

fft_obs = np.array(fft_obs)
PSA_mean_gm0 = np.percentile(fft_gm0, 50, axis=0)
PSA_16_gm0 = np.percentile(fft_gm0, 16, axis=0)
PSA_84_gm0 = np.percentile(fft_gm0, 84, axis=0)

PSA_mean_obs = np.percentile(fft_obs, 50, axis=0)
PSA_16_obs = np.percentile(fft_obs, 16, axis=0)
PSA_84_obs = np.percentile(fft_obs, 84, axis=0)

fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.loglog(fft_freq_gm0, PSA_mean_gm0, color='b', label="Synthetics")
ax.fill_between(fft_freq_gm0, PSA_84_gm0, PSA_16_gm0, color="b", alpha=0.4)

ax.loglog(fft_freq_obs, PSA_mean_obs, color='r', label="Data")
ax.fill_between(fft_freq_obs, PSA_84_obs, PSA_16_obs, color="r", alpha=0.4)

ax.set_xlabel("Frequency (Hz)")
ax.set_title('M '+str(mags)+', '+str(dist_bins[idist[0]])+'-'+str(dist_bins[idist[0]+1])+' km, $N_{obs}$ = '+str(len(st_tqdne_EW[:,0])))
ax.set_ylabel(r'Amplitude (m/s$^2$Hz$^{-1}$)')
ax.set_xlim(0.005, 50)
ax.legend(loc='lower left')
fig.savefig("figures/FFT_"+str(mags)+"_"+str(dist_bins[idist[0]])+"-"+str(dist_bins[idist[0]+1])+".png", dpi=300, bbox_inches="tight")

std_PSA_gm0 = np.std(np.log(fft_gm0), axis=0)
std_PSA_obs = np.std(np.log(fft_obs), axis=0)

fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.semilogx(fft_freq_gm0, std_PSA_gm0, color='b', label="Synthetics")
ax.semilogx(fft_freq_obs, std_PSA_obs, color='r', label="Data")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("std(ln(FAS))")
ax.set_xlim(0.005, 50)
ax.set_ylim(0, 2.5)
ax.legend(loc='upper left')
fig.savefig("figures/stdFAS_"+str(mags)+"_"+str(dist_bins[idist[0]])+"-"+str(dist_bins[idist[0]+1])+".png", dpi=300, bbox_inches="tight")
