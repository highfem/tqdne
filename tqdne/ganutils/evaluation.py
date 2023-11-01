#!/usr/bin/env python3

import itertools
import os

import numpy as np
# import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from tqdne.ganutils.data_utils import get_cond_var_bins, get_waves_real_bin
from tqdne.ganutils.plotting import plot_syn_data_single
from tqdne.ganutils.random_fields import rand_noise
from tqdne.ganutils.synthetic_data import get_synthetic_data


sns.set()
color_palette = sns.color_palette('dark')
colors = [color_palette[3], color_palette[7], color_palette[0], color_palette[1], color_palette[2], color_palette[4], color_palette[5], color_palette[6], color_palette[8], color_palette[9]]
sns.set_palette(sns.color_palette(colors))
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")


def calc_mean_distances(G, dataset, wfs, c_norms, means, samples, noise_dim):
    c_norms = c_norms.reshape(-1, 1)
    real_data = wfs * c_norms
    real_data_mean = np.mean(real_data, axis=0)

    dist = means['dist']
    mag = means['mag']

    dist_max = dataset.df_meta['dist'].max()
    mag_max = dataset.df_meta['mag'].max()

    vc_list = [
        dist / dist_max * torch.ones(samples, 1),
        mag / mag_max * torch.ones(samples, 1),
    ]

    grf = rand_noise(1, noise_dim, device=device)
    random_data = grf.sample(samples)
    syn_data, syn_scaler = G(random_data, *vc_list)
    syn_data = syn_data.squeeze().detach().cpu().numpy()

    syn_data = syn_data * syn_scaler.detach().cpu().numpy()

    sd_mean = np.mean(syn_data, axis=0)

    l2_dist = np.sum(np.abs(real_data_mean - sd_mean)) / len(sd_mean)
    mse = np.sum((real_data_mean - sd_mean)**2) / len(sd_mean)
    # print("L2: ", l2_dist)
    # print("MSE: ", mse)
    return (l2_dist, mse)


def plot_metrics_matrix(G, dataset, vc_bins, dirs, args):
    dist_bin_centers = []
    mag_bin_centers = []

    dist_bins = vc_bins['dist_bins']
    mag_bins = vc_bins['mag_bins']

    n_dist_bins = len(dist_bins) - 1
    n_mag_bins = len(mag_bins) - 1

    for i in range(min(n_dist_bins, n_mag_bins)):
        dist_border = [dist_bins[i], dist_bins[i+1]]
        mag_border = [mag_bins[i], mag_bins[i+1]]
        wfs, c_norms, df_s, means, n_obs = get_waves_real_bin(dataset, dist_border, mag_border)
        dist_bin_centers.append(means['dist'].round(1))
        mag_bin_centers.append(means['mag'].round(1))

    l2_mat = np.full((10, 10), np.nan)
    mse_mat = np.full((10, 10), np.nan)

    dist_max = dataset.df_meta['dist'].max()
    mag_max = dataset.df_meta['mag'].max()

    for i in range(n_dist_bins):
        for j in range(n_mag_bins):
            dist_border = [dist_bins[i], dist_bins[i+1]]
            mag_border = [mag_bins[j], mag_bins[j+1]]
            wfs, c_norms, df_s, means, n_obs = get_waves_real_bin(dataset, dist_border, mag_border)

            if n_obs < 25:
                print(f"Bucket [{i}, {j}] only contains {n_obs} waveforms.")
                print(f"Dist: [{dist_border[0]:.2f}, {dist_border[1]:.2f}], Mag: [{mag_border[0]:.2f}, {mag_border[1]:.2f}]")
                # continue

            # plot_real_syn_bucket(G, wfs, c_norms, means, n_obs, dist_border, mag_border, dirs, dist_max, mag_max, device, args)
            l2, mse = calc_mean_distances(G, dataset, wfs, c_norms, means, n_obs, args.noise_dim)

            l2_mat[i, j] = l2
            mse_mat[i, j] = mse

    plt.title('Log-L2 Distance Real-Synthetic Data')
    ax = sns.heatmap(np.log(l2_mat), xticklabels=mag_bin_centers, yticklabels=dist_bin_centers)
    ax.invert_yaxis()
    plt.xlabel('Bin Mean Magnitude')
    plt.ylabel('Bin Mean Distance [km]')
    fig_file = os.path.join(dirs['metrics_dir'], f"l2_metric_matrix.{args.plot_format}")
    plt.savefig(fig_file, format=f"{args.plot_format}")
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.title('Log-MSE Distance Real-Synthetic Data')
    ax = sns.heatmap(np.log(mse_mat), xticklabels=mag_bin_centers, yticklabels=dist_bin_centers)
    ax.invert_yaxis()
    plt.xlabel('Bin Mean Magnitude')
    plt.ylabel('Bin Mean Distance [km]')
    fig_file = os.path.join(dirs['metrics_dir'], f"mse_metric_matrix.{args.plot_format}")
    plt.savefig(fig_file, format=f"{args.plot_format}")
    plt.close('all')
    plt.clf()
    plt.cla()
    return (l2_mat, mse_mat)


def evaluate_model(G, n_waveforms, dataset, dirs, logger, args):
    fig_dir = dirs['fig_dir']
    metrics_dir = dirs['metrics_dir']
    grid_dir = dirs['grid_dir']
    assert dataset.vc_min["dist"] == dataset.df_meta['dist'].min()
    

    cond_var_bins = get_cond_var_bins(dataset, 10, True)

    n_rows = 8
    n_cols = 4
    n_tot = n_rows * n_cols

    dist_min = dataset.df_meta['dist'].min()
    dist_max = dataset.df_meta['dist'].max()
    dist_mean = dataset.df_meta['dist'].mean()

    mag_min = dataset.df_meta['mag'].min()
    mag_max = dataset.df_meta['mag'].max()
    mag_mean = dataset.df_meta['mag'].mean()

    dists = [dist_min, dist_mean, dist_max]
    mags = [mag_min, mag_mean, mag_max]

    # l2_mat, mse_mat = plot_metrics_matrix(G, dataset, cond_var_bins, dirs, args)
    # mlflow.log_metric(key="l2_total", value=np.sum(l2_mat), step=epoch)
    # mlflow.log_metric(key="mse_total", value=np.sum(mse_mat), step=epoch)
    # mlflow.log_metric(key="l2_avg", value=np.sum(l2_mat) / l2_mat.size, step=epoch)
    # mlflow.log_metric(key="mse_avg", value=np.sum(mse_mat) / mse_mat.size, step=epoch)

    for dist in dists:
        for i, mag in enumerate(cond_var_bins['mag_bins']):
            if i % 2 != 0:
                continue
            vc_list = [
                dist / dist_max * torch.ones(n_waveforms, 1),
                mag / mag_max * torch.ones(n_waveforms, 1),
            ]

            grf = rand_noise(1, args.latent_dim, device=device)
            random_data = grf.sample(n_waveforms)
            syn_data, syn_scaler = G(random_data, *vc_list)
            syn_data = syn_data.squeeze().detach().cpu().numpy()
            syn_data = syn_data * syn_scaler.detach().cpu().numpy()

            synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
            sd_mean = np.mean(synthetic_data_log, axis=0)

            y = np.exp(sd_mean)

            nt = synthetic_data_log.shape[1]
            tt = args.time_delta * np.arange(0, nt)
            plt.semilogy(tt, y, '-' , label=f'Dist: {dist:.2f}km, Mag: {mag:.2f}', alpha=0.8, lw=0.5)

        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Log-Amplitude')
        fig_file = os.path.join(fig_dir, f"syn_data_mag_dependence_dist_{dist:.2f}.{args.plot_format}")
        plt.savefig(fig_file, format=f"{args.plot_format}")
        plt.close('all')
        plt.clf()
        plt.cla()

    for mag in mags:
        for i, dist in enumerate(cond_var_bins['dist_bins']):
            if i % 2 != 0:
                continue
            vc_list = [
                dist / dist_max * torch.ones(n_waveforms, 1),
                mag / mag_max * torch.ones(n_waveforms, 1),
            ]

            grf = rand_noise(1, args.latent_dim, device=device)
            random_data = grf.sample(n_waveforms)
            syn_data, syn_scaler = G(random_data, *vc_list)
            syn_data = syn_data.squeeze().detach().cpu().numpy()
            syn_data = syn_data * syn_scaler.detach().cpu().numpy()

            synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
            sd_mean = np.mean(synthetic_data_log, axis=0)

            y = np.exp(sd_mean)

            nt = synthetic_data_log.shape[1]
            tt = args.time_delta * np.arange(0, nt)
            plt.semilogy(tt, y, '-' , label=f'Dist: {dist:.2f}km, Mag: {mag:.2f}', alpha=0.8, lw=0.5)

        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Log-Amplitude')
        fig_file = os.path.join(fig_dir, f"syn_data_dist_dependence_mag_{mag:.2f}.{args.plot_format}")
        plt.savefig(fig_file, format=f"{args.plot_format}")
        plt.close('all')
        plt.clf()
        plt.cla()

    for dist, mag in itertools.product(dists, mags):
        vc_list = [
                dist / dist_max * torch.ones(n_tot, 1),
                mag / mag_max * torch.ones(n_tot, 1),
        ]
        grf = rand_noise(1, args.latent_dim, device=device)
        random_data = grf.sample(n_tot)
        syn_data, syn_scaler = G(random_data, *vc_list)
        syn_data = syn_data.squeeze()
        syn_data = np.array(syn_data.detach().cpu())
        syn_scaler = syn_scaler.detach().cpu().numpy()

        syn_data = syn_data * syn_scaler

        n_t = syn_data.shape[1]
        tt = args.time_delta * np.arange(0, n_t)

        plt.figure()
        fig, axs = plt.subplots(n_rows, n_cols, sharex='col',
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.15},
                                figsize=(36,12),
                                )

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude", labelpad=10.0)

        for i, ax in enumerate(axs.flat):
            ax.plot(tt, syn_data[i, :], linewidth=0.5)
            low, high = ax.get_ylim()
            bound = max(abs(low), abs(high))
            ax.set_ylim(-bound, bound)

        # plt.tight_layout(pad=0.2)
        fig.suptitle(f'Randomly drawn samples from Generator. Dist: {dist:.2f} km, Mag: {mag:.2f}')
        fig_file = os.path.join(dirs['grid_dir'], f"generated_data_{dist:.2f}_km_mag_{mag:.2f}.{args.plot_format}")
        plt.savefig(fig_file, format=f"{args.plot_format}")
        plt.close('all')
        plt.clf()
        plt.cla()

        # ---------- 3x3 matrices ----------
        plt.figure()
        fig, axs = plt.subplots(3, 3, sharex='col', sharey='all',
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.05},
                                figsize=(20,12),
                                )
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        for i, distbucket in enumerate([0, 4, 8]):
            for j, magbucket in enumerate([1, 5, 8]):
                dist_border = [cond_var_bins['dist_bins'][distbucket], cond_var_bins['dist_bins'][distbucket+1]]
                mag_border = [cond_var_bins['mag_bins'][magbucket], cond_var_bins['mag_bins'][magbucket+1]]
                wfs, c_norms, df_s, means, n_obs = get_waves_real_bin(dataset, dist_border, mag_border)

                c_norms = c_norms.reshape(-1, 1)
                real_data = np.log(np.abs(wfs * c_norms) + 1e-10)
                # rd_25 = np.exp(np.percentile(real_data, 25, axis=0))
                # rd_75 = np.exp(np.percentile(real_data, 75, axis=0))
                real_data_mean = np.exp(real_data.mean(axis=0))
                real_data = np.exp(real_data)

                samples = n_obs
                dt = 0.05

                dist = means['dist']
                mag = means['mag']

                vc_list = [
                    dist / dist_max * torch.ones(samples, 1),
                    mag / mag_max * torch.ones(samples, 1),
                ]

                random_data = grf.sample(samples)
                syn_data, syn_scaler = G(random_data, *vc_list)
                syn_data = syn_data.squeeze().detach().cpu().numpy()
                syn_data = syn_data * syn_scaler.detach().cpu().numpy()
                synthetic_data_log = np.log(np.abs(syn_data + 1e-10))
                sd_mean = np.mean(synthetic_data_log, axis=0)
                sd_mean = np.exp(sd_mean)

                # sd_25 = np.exp(np.percentile(synthetic_data_log, 25, axis=0))
                # sd_75 = np.exp(np.percentile(synthetic_data_log, 75, axis=0))

                Nt = synthetic_data_log.shape[1]
                tt = dt * np.arange(0, Nt)

                # fig = plt.figure(figsize=(16, 8))
                axs[i,j].semilogy(tt, sd_mean, '-' , label=f'Synthetic Data', alpha=0.8, lw=0.5)
                # axs[i,j].fill_between(tt, sd_75, sd_25, alpha=0.1)
                axs[i,j].semilogy(tt, real_data_mean, '-', label=f'Real Data', alpha=0.8, lw=0.5)
                # axs[i,j].fill_between(tt, rd_75, rd_25, alpha=0.1)

                axs[i,j].set_title(f"Obs: {n_obs}, Dist: [{dist_border[0]:.1f},{dist_border[1]:.1f}] km, Mag: [{mag_border[0]:.1f},{mag_border[1]:.1f}]")
                axs[i,j].legend(loc=4)

        axs[1, 0].set_ylabel('Log-Amplitude')
        axs[2, 1].set_xlabel('Time [s]')
        # fig.suptitle(f'Randomly drawn samples from Generator. Dist: {dist:.2f} km, Mag: {mag:.2f}')
        fig_file = os.path.join(dirs['metrics_dir'], f"3x3_model_eval.{args.plot_format}")
        plt.savefig(fig_file, format=f"{args.plot_format}")
        plt.close('all')
        plt.clf()
        plt.cla()

        # --- freq ----
        plt.figure()
        fig, axs = plt.subplots(3, 3, sharex='col', sharey='all',
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.05},
                                figsize=(20,12),
                                )
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        for i, distbucket in enumerate([0, 4, 8]):
            for j, magbucket in enumerate([1, 5, 8]):
                samples = n_obs
                dt = 0.05

                dist = means['dist']
                mag = means['mag']

                vc_list = [
                    dist / dist_max * torch.ones(samples, 1),
                    mag / mag_max * torch.ones(samples, 1),
                ]

                random_data = grf.sample(samples)
                syn_data, syn_scaler = G(random_data, *vc_list)
                syn_data = syn_data.squeeze().detach().cpu().numpy()
                syn_data = syn_data * syn_scaler.detach().cpu().numpy()

                s_fft = np.fft.rfft(syn_data)
                s_ps = np.sqrt(np.real(s_fft * np.conj(s_fft)))
                s_freq = np.fft.rfftfreq(syn_data.shape[1], dt)

                # TODO:
                # Change to mean of log or median!
                sig = np.median(s_ps, axis=0)
                axs[i,j].loglog(s_freq, sig, lw=0.5, label='Synthetic Data')

                sd_25 = np.exp(np.percentile(np.log(s_ps), 25, axis=0))
                sd_75 = np.exp(np.percentile(np.log(s_ps), 75, axis=0))

                nt = sig.shape[0]
                tt = np.linspace(s_freq.min(), s_freq.max(), nt)
                axs[i,j].fill_between(tt, sd_75, sd_25, alpha=0.1)

                dist_border = [cond_var_bins['dist_bins'][distbucket], cond_var_bins['dist_bins'][distbucket+1]]
                mag_border = [cond_var_bins['mag_bins'][magbucket], cond_var_bins['mag_bins'][magbucket+1]]
                wfs, c_norms, df_s, means, n_obs = get_waves_real_bin(dataset, dist_border, mag_border)

                c_norms = c_norms.reshape(-1, 1)
                signal = wfs * c_norms

                s_fft = np.fft.rfft(signal)
                s_ps = np.sqrt(np.real(s_fft * np.conj(s_fft)))
                freq = np.fft.rfftfreq(signal.shape[1], dt)
                sig = np.median(s_ps, axis=0)

                axs[i,j].loglog(freq, sig, lw=0.5, label='Real Data')

                # real_data = np.log(np.abs(wfs * c_norms) + 1e-10)
                rd_25 = np.exp(np.percentile(np.log(s_ps), 25, axis=0))
                rd_75 = np.exp(np.percentile(np.log(s_ps), 75, axis=0))

                nt = sig.shape[0]
                tt = np.linspace(freq.min(), freq.max(), nt)
                axs[i,j].fill_between(tt, rd_75, rd_25, alpha=0.1)

                axs[i,j].set_title(f"Obs: {n_obs}, Dist: [{dist_border[0]:.1f},{dist_border[1]:.1f}] km, Mag: [{mag_border[0]:.1f},{mag_border[1]:.1f}]")
                axs[i,j].legend(loc=2)

        axs[1, 0].set_ylabel('Log Fourier Amplitude')
        axs[2, 1].set_xlabel('Frequency [Hz]')
        fig_file = os.path.join(dirs['metrics_dir'], f"3x3_model_eval_frequency.{args.plot_format}")
        plt.savefig(fig_file, format=f"{args.plot_format}")
        plt.close('all')
        plt.clf()
        plt.cla()

        samples = get_synthetic_data(
            G,
            n_waveforms,
            dataset,
            dist,
            mag,
            args
        )

        plot_syn_data_single(samples, dist, mag, fig_dir, args)

    logger.log_artifacts(f"{dirs['fig_dir']}", f"{dirs['output_dir']}/figs")
    logger.log_artifacts(f"{dirs['metrics_dir']}", f"{dirs['output_dir']}/metrics")
    logger.log_artifacts(f"{dirs['grid_dir']}", f"{dirs['output_dir']}/grid_plots")
