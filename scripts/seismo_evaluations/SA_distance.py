import glob

import matplotlib.pyplot as plt
import numpy as np
from example_GMM import calculate_gmfs, parallel_processing_sa
from utils import SeismicParameters

# plt.rc('text', usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 16})

dataset_path = glob.glob("../experiments/workdir/h5_generativeGMM/for_GenerativeGMM*.h5")
dataset_path = np.sort(dataset_path)
for ii in range(len(dataset_path)):
    # for i in range(1):
    print(dataset_path[ii])
    files_1 = SeismicParameters(dataset_path[ii])

    waveforms = files_1.waveforms
    rhyp = files_1.hypocentral_distance
    vs30 = np.mean(files_1.vs30s)
    mags = np.mean(files_1.magnitude)

    angles = np.arange(0, 180, step=1)
    dt = 0.01
    T = 0.1
    g_inv = 1 / 9.805
    wf_NS = waveforms[:, 0, :]
    wf_EW = waveforms[:, 1, :]
    periods = np.array([0.1, 0.3, 1.0, 2.0])
    percentile = 50

    sa_mean_01 = parallel_processing_sa(wf_NS, wf_EW, dt, periods, percentile)
    sa_mean_01 = np.array(sa_mean_01)

    files = SeismicParameters("../experiments/workdir/data/preprocessed_waveforms.h5")
    rhyp_obs = files.hypocentral_distance * 1e3
    mag_obs = files.magnitude
    vs30_obs = files.vs30s
    wf_obs = files.waveforms
    mask1 = (vs30_obs <= vs30 + 50) & (vs30_obs > vs30 - 50)
    mask2 = (mag_obs <= mags + 0.1) & (mag_obs > mags - 0.1)
    masked = mask1 & mask2
    wf_EW_obs = wf_obs[masked, 1, :]
    wf_NS_obs = wf_obs[masked, 0, :]
    rhyp_obs = rhyp_obs[masked]

    sa_mean_obs = parallel_processing_sa(wf_NS_obs, wf_EW_obs, dt, periods, percentile)
    sa_mean_obs = np.array(sa_mean_obs)

    SA_type = ["SA(0.1)", "SA(0.3)", "SA(1.0)", "SA(2.0)"]
    for j in range(4):
        mag = mags
        rupture_aratio = 1.5
        strike = 45
        dip = 50
        rake = 0
        lon = 9.1500
        lat = 45.1833
        depth = 15
        Vs30 = np.mean(vs30)
        hypocenter = [lon, lat, depth]
        imts = [SA_type[j]]
        gmpes = ["BooreEtAl2014", "Kanno2006Shallow"]

        gms, jb_distances = calculate_gmfs(
            mag, rupture_aratio, strike, dip, rake, hypocenter, imts, Vs30, gmpes
        )
        idx = jb_distances >= 1

        bin_plot = np.linspace(0.1, 190, 100)
        dist_filt = []
        sa_mean = []
        sa_16 = []
        sa_84 = []
        period = j
        for ik in range(len(bin_plot) - 1):
            mask = (rhyp > bin_plot[ik]) & (rhyp <= bin_plot[ik + 1])
            dist_filt.append(0.5 * (bin_plot[ik] + bin_plot[ik + 1]))
            sa_mean.append(np.percentile(sa_mean_01[mask, period], 50))
            sa_16.append(np.percentile(sa_mean_01[mask, period], 16))
            sa_84.append(np.percentile(sa_mean_01[mask, period], 84))

        bin_plot2 = np.linspace(1, 190, 20)
        rhyp_filt2 = []
        sa_mean2 = []
        sa_162 = []
        sa_842 = []
        std = []
        for il in range(len(bin_plot2) - 1):
            mask = (rhyp_obs > bin_plot2[il]) & (rhyp_obs <= bin_plot2[il + 1])
            if len(np.where(mask == True)[0]) > 0:
                rhyp_filt2.append(0.5 * (bin_plot2[il] + bin_plot2[il + 1]))
                sa_mean2.append(np.percentile(sa_mean_obs[mask, period], 50))
                sa_162.append(np.percentile(sa_mean_obs[mask, period], 16))
                sa_842.append(np.percentile(sa_mean_obs[mask, period], 84))
                std.append(np.std(sa_mean_obs[mask, period]))
            else:
                continue

        plt.rcParams.update({"font.size": 16})

        print("Plotting figures...")

        fig = plt.figure(figsize=(7, 4))
        ax = fig.gca()
        colors = ["#0072B2", "#F0E442"]  # plt.cm.rainbow(np.linspace(0, 1, len(gmpes)))
        for im in range(len(rhyp_filt2)):
            ax.scatter(
                rhyp_filt2[im],
                sa_mean2[im],
                marker="s",
                s=40,
                color="grey",
                alpha=0.9,
                label="Data-bin med.",
            )
            ax.plot(
                [rhyp_filt2[im], rhyp_filt2[im]],
                [sa_mean2[im], sa_162[im]],
                "-",
                color="grey",
                alpha=0.9,
            )
            ax.plot(
                [rhyp_filt2[im], rhyp_filt2[im]],
                [sa_mean2[im], sa_842[im]],
                "-",
                color="grey",
                alpha=0.9,
            )
        for imt in range(len(imts)):
            for gsi, color in zip(range(len(gmpes)), colors):
                ax.plot(
                    jb_distances[idx],
                    np.exp(gms[0, gsi, imt][idx]),
                    linewidth=2,
                    color=color,
                    label=gmpes[gsi],
                )
                ax.plot(
                    jb_distances[idx],
                    np.exp(gms[0, gsi, imt][idx] + gms[1, gsi, imt][idx]),
                    "--",
                    linewidth=1,
                    color=color,
                    alpha=0.9,
                )
                ax.plot(
                    jb_distances[idx],
                    np.exp(gms[0, gsi, imt][idx] - gms[1, gsi, imt][idx]),
                    "--",
                    linewidth=1,
                    color=color,
                    alpha=0.9,
                )
        if sa_mean_obs.shape[0] > 0:
            ax.scatter(
                rhyp_obs,
                sa_mean_obs[:, period],
                color="k",
                s=4,
                linewidth=0.2,
                edgecolor="k",
                label="Data",
            )
            ax.set_title(
                f"V$_{{S30}}$: {vs30 - 50}-{vs30 + 50} m/s,  M{mags-0.1}-{mags+0.1}, T = {periods[period]}s, N$_{{obs}}$: {sa_mean_obs[:,period].shape[0]}",
                fontsize=14,
            )
        else:
            ax.set_title(
                f"V$_{{S30}}$: {vs30 - 50}-{vs30 + 50} m/s, M{mags-0.1}-{mags+0.1}, T = {periods[period]}s, N$_{{obs}}$: 0",
                fontsize=14,
            )
        ax.plot(dist_filt, sa_mean, "-", color="r", markersize=7, label="GWM-med.")
        ax.fill_between(dist_filt, sa_16, sa_84, color="r", alpha=0.2, label="GWM-std.")
        ax.set_xlabel("Hypocentral Distance [km]")
        ax.set_ylabel(f"SA({periods[period]}s) [m/s$^2$]")
        ax.set_yscale("log")
        # ax.set_xscale('log')
        # ax.legend()
        ax.grid(axis="both", which="both", linewidth=0.5)
        ax.set_xlim(1, 190)
        ax.set_ylim(1e-6, 10)
        fig.savefig(
            f"figures/SA/nolog_new/SA_distance_Vs30{vs30}_M{mags}_T{periods[period]}s_nolog.png",
            dpi=200,
            bbox_inches="tight",
        )
