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
from glob import glob

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

# divide samples in dist_bins
dist_bins_samples = []
waveforms_samples = []

max_num = 1000
for i in range(1):
    mask2 = (features[:, 2] >= mag_bins[imag]) & (features[:, 2] < mag_bins[imag + 1])
    mask3 = (features[:, 3] > 0)
    mask = mask2 & mask3
    waveforms_samples.append(np.nan_to_num((waveforms[mask][:max_num])))
    dist_bins_samples.append(np.nan_to_num((features[mask,0][:max_num])))
# magnitude
mags = np.round((mag_bins[imag]+mag_bins[imag+1])/2, decimals=1)

dataset_folder = 'GM0_waveforms/'+str(mags)+'/'
data_list = glob(dataset_folder+'/*.pkl')
wf = []
conditional = []
for dat in data_list:
    with open(dat, 'rb') as f:
        shakemap_data_gen = pickle.load(f)
        print(shakemap_data_gen['waveforms'].shape, shakemap_data_gen['cond'].shape)

    conditional_init = shakemap_data_gen['cond']
    wf_init = shakemap_data_gen['waveforms']
    wf.append(wf_init)
    conditional.append(conditional_init)
conditional = np.vstack(conditional)
wf = np.vstack(wf)
print(conditional.shape, wf.shape)
rhyp = conditional[:,0]
is_shallow_crustal = conditional[:,1]
magnitude = conditional[:,2]
vs30 = conditional[:,3]
ave_vs30 = np.mean(vs30)
ave_magnitude = np.mean(magnitude)

dt = 0.01
g_inv = 1/9.80665
st_tqdne_EW = wf[:,1,:]
st_tqdne_NS = wf[:,0,:]

wf_EW = waveforms_samples[0][:,1,:]
wf_NS = waveforms_samples[0][:,0,:]

### PGA
import pyrotd
import obspy as ob
from obspy.core.inventory.inventory import read_inventory

## PGA GMRotD50 max
if False:
    PGA_geom_mean_obs = []
    PGA_geom_mean_gm0 = []
    g_inv = 1/9.80665
    for i in range(len(wf_EW[:,0])):
        EW_obs = wf_EW[i, :]*g_inv
        NS_obs = wf_NS[i, :]*g_inv
        gmrot50 = calculate_gmrotd50(EW_obs, NS_obs)
        PGA_geom_mean_obs = np.append(PGA_geom_mean_obs, gmrot50)
    for ii in range(len(st_tqdne_EW[:,0])):
        # GM0
        EW_gm0 = st_tqdne_EW[ii,:]*g_inv
        NS_gm0 = st_tqdne_NS[ii,:]*g_inv
        gmrot50_gm0 = calculate_gmrotd50(EW_gm0, NS_gm0)
        PGA_geom_mean_gm0 = np.append(PGA_geom_mean_gm0, gmrot50_gm0)
    
    mag = ave_magnitude
    rupture_aratio = 1.5
    strike = 236
    dip = 51
    rake = 110
    lon = 137.89
    lat = 36.69
    depth = 9
    hypocenter = [lon, lat, depth]
    imts = ['PGA']
    gmpes = ['BooreEtAl2014', 'Kanno2006Shallow']
    gms, jb_distances = calculate_gmfs(mag, rupture_aratio, strike, dip, rake, hypocenter, imts, gmpes)


    # plot distance scaling
    fig = plt.figure(figsize=(7,7))
    ax = fig.gca()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(gmpes)))
    ax.loglog(dist_bins_samples[0], PGA_geom_mean_obs, "o", color='grey', markersize=5, markeredgecolor='k', markeredgewidth=0.5, label="Observation")
    ax.loglog(rhyp, PGA_geom_mean_gm0, "^", color='blue', markersize=5, markeredgecolor='k', markeredgewidth=0.5, label="GM0")
    for imt in range(len(imts)):
            for gsi, color in zip(range(len(gmpes)), colors):
                ax.loglog(jb_distances, np.exp(gms[0, gsi, imt]), linewidth=2, color=color, label=gmpes[gsi])
                ax.loglog(jb_distances, np.exp(gms[0, gsi, imt] + gms[1, gsi, imt]), '--', linewidth=1, color=color, alpha=0.6)
                ax.loglog(jb_distances, np.exp(gms[0, gsi, imt] - gms[1, gsi, imt]), '--', linewidth=1, color=color, alpha=0.6)
    for aa in range(len(dist_bins)-1):
        idx = (rhyp >= dist_bins[aa]) & (rhyp < dist_bins[aa + 1])
        mid = 0.5*(dist_bins[aa] + dist_bins[aa+1])
        mean_PGA_bin = np.mean(PGA_geom_mean_gm0[idx])
        std = np.std(PGA_geom_mean_gm0[idx])
        ax.loglog(mid, mean_PGA_bin, 's', markersize=6, color='k')
        ax.errorbar(mid, mean_PGA_bin, yerr=std, elinewidth=2, ecolor='k', capsize=2)
    ax.set_xlabel('Hypocentral Distance (km)')
    ax.set_ylabel('PGA (g)')
    ax.legend(fontsize=12, frameon=False)
    ax.set_xlim(1, 350)
    fig.savefig("figures/PGA_obs_GMM.png", dpi=300, bbox_inches="tight")
    # plt.show()

## PGV GMRotD50 max
if True:
    # Integrate the acceleration
    PGV_geom_mean_obs = []
    PGV_geom_mean_gm0 = []
    for i in range(len(wf_EW[:,0])):
        wf_EW_vel = it.cumtrapz(wf_EW[i, :], dx=dt)
        wf_NS_vel = it.cumtrapz(wf_NS[i, :], dx=dt)
        EW_obs = wf_EW_vel*100
        NS_obs = wf_EW_vel*100
        gmrot50 = calculate_gmrotd50(EW_obs, NS_obs)
        PGV_geom_mean_obs = np.append(PGV_geom_mean_obs, gmrot50)
    for aa in range(len(st_tqdne_EW[:,0])):
        # GM0
        EW_gm0 = it.cumtrapz(st_tqdne_EW[aa], dx=dt)*100
        NS_gm0 = it.cumtrapz(st_tqdne_NS[aa], dx=dt)*100
        gmrot50_gm0 = calculate_gmrotd50(EW_gm0, NS_gm0)
        PGV_geom_mean_gm0 = np.append(PGV_geom_mean_gm0, gmrot50_gm0)

    mag = ave_magnitude
    rupture_aratio = 1.5
    strike = 236
    dip = 51
    rake = 110
    lon = 137.89
    lat = 36.69
    depth = 9
    hypocenter = [lon, lat, depth]
    imts = ['PGV']
    gmpes = ['BooreEtAl2014', 'Kanno2006Shallow']
    gms, jb_distances = calculate_gmfs(mag, rupture_aratio, strike, dip, rake, hypocenter, imts, gmpes)


    # plot distance scaling
    fig = plt.figure(figsize=(7,7))
    ax = fig.gca()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(gmpes)))
    ax.loglog(dist_bins_samples[0], PGV_geom_mean_obs, "o", color='grey', markersize=5, markeredgecolor='k', markeredgewidth=0.5, label="Observation")
    ax.loglog(rhyp, PGV_geom_mean_gm0, "^", color='blue', markersize=5, markeredgecolor='k', markeredgewidth=0.5, label="GM0")
    for imt in range(len(imts)):
            for gsi, color in zip(range(len(gmpes)), colors):
                ax.loglog(jb_distances, np.exp(gms[0, gsi, imt]), linewidth=2, color=color, label=gmpes[gsi])
                ax.loglog(jb_distances, np.exp(gms[0, gsi, imt] + gms[1, gsi, imt]), '--', linewidth=1, color=color, alpha=0.6)
                ax.loglog(jb_distances, np.exp(gms[0, gsi, imt] - gms[1, gsi, imt]), '--', linewidth=1, color=color, alpha=0.6)
    for aa in range(len(dist_bins)-1):
        idx = (rhyp >= dist_bins[aa]) & (rhyp < dist_bins[aa + 1])
        mid = 0.5*(dist_bins[aa] + dist_bins[aa+1])
        mean_PGV_bin = np.mean(PGV_geom_mean_gm0[idx])
        std = np.std(PGV_geom_mean_gm0[idx])
        ax.loglog(mid, mean_PGV_bin, 's', markersize=6, color='k')
        ax.errorbar(mid, mean_PGV_bin, yerr=std, elinewidth=2, ecolor='k', capsize=2)
    ax.set_xlabel('Hypocentral Distance (km)')
    ax.set_ylabel('PGV (cm/s)')
    ax.legend(fontsize=12, frameon=False, loc='lower left')
    ax.set_xlim(1, 350)
    fig.savefig("figures/PGV_obs_GMM.png", dpi=300, bbox_inches="tight")


