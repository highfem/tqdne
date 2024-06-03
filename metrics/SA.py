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

PSA_resp_gm0_50 = []
for i in range(len(st_tqdne_EW[:,0])):
    EW_gm0 = st_tqdne_EW[i,:]*g_inv
    NS_gm0 = st_tqdne_NS[i,:]*g_inv
    time_step = 0.01
    osc_damping = 0.05
    osc_freqs = np.logspace(-1, 2, 200)
    osc_periods = 1/osc_freqs   

    rotated_resp = pyrotd.calc_rotated_spec_accels(
            time_step, 
            EW_gm0, NS_gm0,
            osc_freqs, osc_damping, percentiles=[50],
    )
    selected = rotated_resp[rotated_resp.percentile == 50]
    PSA_resp_gm0_50.append(selected.spec_accel)
PSA_resp_gm0_50 = np.array(PSA_resp_gm0_50)

# finding index of binning observation
PSA_resp_obs_50 = []
for i in range(len(wf_EW[:,0])):
    EW_obs = wf_EW[i,:]*g_inv
    EW_obs[np.isnan(EW_obs)] = 0
    NS_obs = wf_NS[i,:]*g_inv
    NS_obs[np.isnan(NS_obs)] = 0
    time_step = 0.01
    osc_damping = 0.05
    osc_freqs = np.logspace(-1, 2, 91)
    osc_periods = 1/osc_freqs   

    rotated_resp_obs = pyrotd.calc_rotated_spec_accels(
            time_step, 
            EW_obs, NS_obs,
            osc_freqs, osc_damping, percentiles=[50],
    )
    selected_obs = rotated_resp_obs[rotated_resp_obs.percentile == 50]
    PSA_resp_obs_50.append(selected_obs.spec_accel)
PSA_resp_obs_50 = np.array(PSA_resp_obs_50)

PSA_mean_gm0 = np.percentile(PSA_resp_gm0_50, 50, axis=0)
PSA_16_gm0 = np.percentile(PSA_resp_gm0_50, 16, axis=0)
PSA_84_gm0 = np.percentile(PSA_resp_gm0_50, 84, axis=0)

PSA_mean_obs = np.percentile(PSA_resp_obs_50, 50, axis=0)
PSA_16_obs = np.percentile(PSA_resp_obs_50, 16, axis=0)
PSA_84_obs = np.percentile(PSA_resp_obs_50, 84, axis=0)

fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.loglog(1/selected.osc_freq, PSA_mean_gm0, color='b', label="Synthetics")
ax.fill_between(1/selected.osc_freq, PSA_84_gm0, PSA_16_gm0, color="b", alpha=0.4)

ax.loglog(1/selected_obs.osc_freq, PSA_mean_obs, color='r', label="Data")
ax.fill_between(1/selected_obs.osc_freq, PSA_84_obs, PSA_16_obs, color="r", alpha=0.4)

ax.set_xlabel("Period (s)")
ax.set_title('M '+str(mags)+', '+str(dist_bins[idist[0]])+'-'+str(dist_bins[idist[0]+1])+' km, $N_{obs}$ = '+str(len(st_tqdne_EW[:,0])))
ax.set_ylabel(r'5$\%$-Damped SARotD50 (g)')
ax.set_xlim(0.01, 10)
ax.legend(loc='lower left')
fig.savefig("figures/SA_"+str(mags)+"_"+str(dist_bins[idist[0]])+"-"+str(dist_bins[idist[0]+1])+".png", dpi=300, bbox_inches="tight")

std_PSA_gm0 = np.std(np.log(PSA_resp_gm0_50), axis=0)
std_PSA_obs = np.std(np.log(PSA_resp_obs_50), axis=0)

fig = plt.figure(figsize=(10,5))
ax = fig.gca()
ax.semilogx(1/selected.osc_freq, std_PSA_gm0, color='b', label="Synthetics")
ax.semilogx(1/selected_obs.osc_freq, std_PSA_obs, color='r', label="Data")
ax.set_xlim(0.01, 10)
ax.set_ylim(0, 2.5)
ax.legend(loc='upper left')
fig.savefig("figures/stdSA_"+str(mags)+"_"+str(dist_bins[idist[0]])+"-"+str(dist_bins[idist[0]+1])+".png", dpi=300, bbox_inches="tight")

