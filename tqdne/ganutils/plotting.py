#!/usr/bin/env python3

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from tqdne.ganutils.random_fields import rand_noise


sns.set()
color_palette = sns.color_palette('dark')
colors = [color_palette[3], color_palette[7], color_palette[0], color_palette[1], color_palette[2], color_palette[4], color_palette[5], color_palette[6], color_palette[8], color_palette[9]]
sns.set_palette(sns.color_palette(colors))
# mpl.use('Agg')

def _rolling_window(x, window_len, step_length):
    pos = 0
    while pos <= len(x) - window_len:
        yield x[pos : pos+window_len]
        pos += step_length


def plot_syn_data_single(samples, dist, mag, fig_dir, args):
    # TODO: Rethink if the dependency injecton of having the model
    # discriminator size being the signal length is sound
    tt = args.time_delta * np.arange(args.discriminator_size)

    for j in range(5):
        log_signal = np.log(np.abs(np.array(samples[j])))
        signal = np.array(samples[j])

        fig, ax = plt.subplots(5, 1, figsize=(15, 15))

        ax[0].plot(tt, signal, lw=0.5, label='Signal')
        low, high = ax[0].get_ylim()
        bound = max(abs(low), abs(high))
        ax[0].set_ylim(-bound, bound)
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Amplitude')
        ax[0].legend()

        # Make window length 1 s long
        window_length = int(1 / args.time_delta)
        half_wl = int(window_length / 2)
        
        rw = _rolling_window(log_signal, window_length, window_length)
        rolling_mean = []
        for i, window in enumerate(rw):
            rolling_mean.append(window.mean())
        
        locs = args.time_delta * np.arange(half_wl, len(rolling_mean) * window_length, window_length)

        """
        rw = _rolling_window(log_signal, window_length, 1)
        causal_mean = []
        for i, window in enumerate(rw):
            causal_mean.append(np.median(window))
        ax[1].plot(tt[window_length:], causal_mean, label='Causal Mean')
        """
        
        ax[1].plot(tt, log_signal, lw=0.5, label='Log Transformed Signal')
        ax[1].plot(locs, rolling_mean, label='Centered Mean')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Log Amplitude')
        ax[1].legend()

        rolling_max = []
        rw = _rolling_window(log_signal, window_length, window_length)

        for i, window in enumerate(rw):
            rolling_max.append(window.max())

        ax[2].plot(tt, log_signal, lw=0.5, label='Log Transformed Signal')
        ax[2].plot(locs, rolling_max, label='Centered Max')
        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylabel('Log Amplitude')
        ax[2].legend()

        rolling_max = [None] * (window_length -1)
        rw = _rolling_window(log_signal, window_length, 1)
        for window in rw:
            rolling_max.append(window.max())

        ax[3].plot(tt, log_signal, lw=0.5, label='Log Transformed Signal')
        ax[3].plot(tt, rolling_max, label='Causal Rolling Max')
        ax[3].set_xlabel('Time [s]')
        ax[3].set_ylabel('Log Amplitude')
        ax[3].legend()

        s_fft = np.fft.rfft(signal)
        s_ps = np.real(s_fft * np.conj(s_fft))
        freq = np.fft.rfftfreq(signal.size, d=args.time_delta)

        rw = _rolling_window(s_ps, window_length, window_length)
        rolling_mean = []

        for i, window in enumerate(rw):
            rolling_mean.append(window.mean())

        locs_mean = np.linspace(freq.min(), freq.max(), 25)
        ax[4].loglog(freq, s_ps, lw=0.5, label='Fourier Transformed Signal')
        ax[4].plot(locs_mean, rolling_mean, label='Centered Mean')
        ax[4].set_xlabel('Frequency [Hz]')
        ax[4].set_ylabel('Fourier Amplitude')
        ax[4].set_xlim((0.1, 50))
        ax[4].legend()

        fig.suptitle(f'Random Synthetic Waveform. Dist: {dist:.2f}, Mag: {mag:.2f}')
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout()

        fig_file = os.path.join(fig_dir, f'synthetic_data_plot_{dist:.2f}_km_mag_{mag:.2f}.{args.plot_format}')
        plt.savefig(fig_file, format=f'{args.plot_format}')
        plt.close('all')
        plt.clf()
        plt.cla()


def plot_syn_data_grid(samples, dist, mag, fig_dir, args):
    n_rows = 12
    n_cols = 6
    n_tot = n_rows * n_cols 

    n_t = np.shape(samples[0])[0]
    tt = args.time_delta * np.arange(0, n_t)

    plt.figure()
    fig, axs = plt.subplots(n_rows, n_cols, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.2, 'wspace': 0.05},
                            figsize=(20,12),
                            )
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    for i, ax in enumerate(axs.flat):
        ax.set_ylim((-1, 1))
        ax.plot(tt, samples[i], linewidth=0.3)

    fig.suptitle(f'Randomly drawn samples from Generator. Dist: {dist:.2f} km, Mag: {mag:.2f}.')
    fig_file = os.path.join(fig_dir, f'synthetic_data_grid_plot_dist_{dist:.2f}_mag_{mag:.2f}.{args.plot_format}')
    plt.savefig(fig_file, format=f'{args.plot_format}')
    plt.close('all')
    plt.clf()
    plt.cla()

    
def plot_real_syn_bucket(G, wfs, c_norms, means, n_obs, dist_border, mag_border, dirs, dist_max, mag_max, device, args):
    # wfs, c_norms, means, n_obs = get_waves_real_bin(sdat_train, dist_border, mag_border, verbose=3)

    c_norms = c_norms.reshape(-1, 1)
    real_data = np.log(np.abs(wfs * c_norms) + 1e-10)

    rd_25 = np.exp(np.percentile(real_data, 25, axis=0))
    rd_75 = np.exp(np.percentile(real_data, 75, axis=0))

    real_data_mean = np.exp(real_data.mean(axis=0))
    # real_data_median = np.exp(np.median(real_data, axis=0))
    real_data = np.exp(real_data)

    samples = n_obs
    dt = 0.05

    dist = means['dist']
    mag = means['mag']

    vc_list = [
        dist / dist_max * torch.ones(samples, 1).cuda(),
        mag / mag_max * torch.ones(samples, 1).cuda(),
    ]

    grf = rand_noise(1, args.noise_dim, device=device)
    random_data = grf.sample(samples)
    syn_data, syn_scaler = G(random_data, *vc_list)
    syn_data = syn_data.squeeze().detach().cpu().numpy()
    syn_data = syn_data * syn_scaler.detach().cpu().numpy()
    synthetic_data_log = np.log(np.abs(syn_data + 1e-10))
    sd_mean = np.mean(synthetic_data_log, axis=0)
    sd_mean = np.exp(sd_mean)

    sd_25 = np.exp(np.percentile(synthetic_data_log, 25, axis=0))
    sd_75 = np.exp(np.percentile(synthetic_data_log, 75, axis=0))

    Nt = synthetic_data_log.shape[1]
    tt = dt * np.arange(0, Nt)

    fig = plt.figure(figsize=(16, 8))
    plt.semilogy(tt, sd_mean, '-' , label=f'Synthetic Data', alpha=0.8, lw=0.5)
    plt.fill_between(tt, sd_75, sd_25, alpha=0.1)
    plt.semilogy(tt, real_data_mean, '-' , label=f'Real Data', alpha=0.8, lw=0.5)
    # plt.semilogy(tt, real_data_median, '-' , label=f'Real Data Median', alpha=0.8, lw=0.5)
    plt.fill_between(tt, rd_75, rd_25, alpha=0.1)

    plt.ylabel('Log-Amplitude')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title(f"Obs: {n_obs}, Dist: [{dist_border[0]:.1f},{dist_border[1]:.1f}] km, Mag: [{mag_border[0]:.1f},{mag_border[1]:.1f}]")
    fig_file = os.path.join(dirs['metrics_dir'], f'synthetic_real_comp_dist_{dist_border[0]:03.0f}-{dist_border[1]:03.0f}_mag_{mag_border[0]:02.1f}-{mag_border[1]:02.1f}.{args.plot_format}')
    plt.savefig(fig_file, format=f'{args.plot_format}')
    plt.close('all')
    plt.clf()
    plt.cla()


def plot_waves_1C(dataset, ws, i_vg, args, t_max=50.0, ylim=None, fig_file=None, n_plots=11, fig_size =(10,15), stitle=None, show_fig=False):
    """
        Make plot of waves
        shape of the data has the form -> (Nobs, Nt)
        :param ws_p: numpy.ndarray of shape (Nobs, Nt)
        :param np: number of 3C waves to show
        :param color: Color of the plot
        :param t_max: max time for the plot in sec
        :return: show a panel with the waves

        """
    nt = int(t_max / args.time_delta)
    if ws.shape[0] < n_plots:
        # in case there are less observations than points to plot
        n_plots = ws.shape[0]

    idx = np.linspace(0, args.batch_size-1, n_plots, dtype=int)
    ws_p = ws[idx, :]

    # select waves to plot
    tt = args.time_delta * np.arange(nt)
    plt.figure()
    fig, ax = plt.subplots(n_plots, 1, sharex='col', 
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.05},
                            figsize=fig_size)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time [s]")

    for ik in range(n_plots):
        wf = ws_p[ik, :]

        dist = dataset.to_real(i_vg[0][idx[ik]], 'dist').cpu().numpy()[0]
        mag = dataset.to_real(i_vg[1][idx[ik]], 'mag').cpu().numpy()[0]

        ax[ik].plot(tt, wf, lw=0.5)

        low, high = ax[ik].get_ylim()
        bound = max(abs(low), abs(high))
        ax[ik].set_ylim(-bound, bound)
        ax[ik].set_title(f'Dist: {dist:.1f}, Mag: {mag:.1f}')

        if ylim is not None:
            ax[ik].set_ylim(ylim)

    # TODO: Fix magic number 5 (middle plot of default val n_plots: 11) 
    ax[5].set_ylabel('Amplitude')

    if stitle is not None:
        fig.suptitle(stitle, fontsize=12)

    if show_fig:
        fig.show()

    if fig_file is not None:
        fformat = fig_file.split('.')[-1]
        print('saving:', fformat)
        fig.savefig(fig_file, format=fformat)
        if not show_fig:
            plt.clf()
    else:
        fig.show()
    plt.close('all')
    plt.clf()
    plt.cla()
