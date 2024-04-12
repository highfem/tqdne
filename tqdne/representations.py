from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from matplotlib import pyplot as plt
import numpy as np

from scipy.signal import hilbert
import torch

from tqdne.conf import Config


@staticmethod
def to_numpy(x):
    if isinstance(x, Sequence):
        return x.__class__(to_numpy(v) for v in x)
    elif isinstance(x, Mapping):
        return x.__class__((k, to_numpy(v)) for k, v in x.items())
    else:
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x


class Representation(ABC):

    def __init__(self, config=Config()):
        self.config = config

    def __str__(self):
        return f"{self.__class__.__name__} - {self.parameters}"

    def get_representation(self, signal):
        #return self._get_representation(np.nan_to_num(to_numpy(signal), nan=0))
        # dataset should not have nan values
        return self._get_representation(to_numpy(signal))

    @abstractmethod
    def _get_representation(self, signal):
        pass

    def invert_representation(self, representation):
        #return self._invert_representation(np.nan_to_num(to_numpy(representation), nan=0))
        # model shouldn't return nan values
        return self._invert_representation(to_numpy(representation))

    @abstractmethod
    def _invert_representation(self, representation):
        pass

    def get_shape(self, signal_input_shape):
        return self.get_representation(torch.zeros(signal_input_shape)).shape
    
    @abstractmethod
    def _plot(self, raw_waveform, title, inverted_waveform=None):
        pass

    def plot(self, raw_waveform, title, inverted_waveform=None):
        return self._plot(raw_waveform, title, inverted_waveform)
    
    @abstractmethod
    def _plot_representation(self, channel=0):
        pass

    def plot_representation(self, channel=0): 
        return self._plot_representation(channel)   

    @abstractmethod
    def _test(self, waveforms):
        pass
    
    @staticmethod
    def test(repr, waveforms):
        repr._test(repr, waveforms)

    def _test_inversion(self, waveforms):
        assert np.allclose(waveforms, self.invert_representation(self.get_representation(waveforms)), atol=1e-6) 

    def update_stats(self, config: Config):
        return self._update_stats(config)

    @abstractmethod
    def _update_stats(self, config: Config):
        pass



class SignalWithEnvelope(Representation):

    def __init__(self, env_function, env_function_params, env_transform, env_transform_params, scaling, config = Config()):
        super().__init__(config)
        self.env_function = self._get_env_function_by_name(env_function)
        self.env_function_params = env_function_params
        self.trans_function, self.inv_trans_function = self._get_trans_function_by_name(env_transform)
        self.env_transform_params = env_transform_params
        self.scaling = scaling

        if self.scaling["type"] == "normalize":
            if self.scaling["scalar"]:
                # Signal statistics for normalization
                # Assuming that maximum of the envelope coincides with the maximum of the signal and that the minimum of the envelope is 0.
                dataset_stats_dict = config.signal_statistics 
                max_env_peak_per_channel = np.array([dataset_stats_dict[ch]['max'] for ch in dataset_stats_dict.keys()])[:, np.newaxis]
                min_env_peak_per_channel = np.array([0 for _ in max_env_peak_per_channel])[:, np.newaxis]
                max_trans_env_peak_per_channel = self.trans_function(max_env_peak_per_channel, **env_transform_params)
                min_trans_env_peak_per_channel = self.trans_function(min_env_peak_per_channel, **env_transform_params)
                #Normalize the transformed envelope to the range [-1, 1]
                self.scaling_function = lambda trans_env: 2 * (trans_env - min_trans_env_peak_per_channel) / (max_trans_env_peak_per_channel - min_trans_env_peak_per_channel) - 1
                self.inv_scaling_function = lambda norm_trans_env: (norm_trans_env + 1) * (max_trans_env_peak_per_channel - min_trans_env_peak_per_channel) / 2 + min_trans_env_peak_per_channel
            else:
                raise NotImplementedError("Channel-wise normalization is not implemented yet -- missing statistics")

        elif self.scaling["type"] == "standardize":
            if self.scaling["scalar"]:
                raise NotImplementedError("Scalar standardization is not implemented yet -- missing statistics")
            else:
                # Statistics of the transformed envelope (computer over a subset of the training dataset)
                # TODO: manage. maybe remove it, as one should need to compute the statistics over the whole dataset for each envelope function and related params
                import logging 
                logging.warning("To be used only with hilbert envelope function and log transformation with eps=1e-5!.")
                env_stats_per_channel = [dataset_stats_dict[ch] for ch in dataset_stats_dict.keys()]
                trans_env_mean_per_channel = np.array([env_stats['mean'] for env_stats in env_stats_per_channel]) # shape: (num_channels, signal_length)
                trans_env_std_per_channel = np.array([env_stats['std_dev'] for env_stats in env_stats_per_channel]) # shape: (num_channels, signal_length)
                # Standardize the transformed envelope 
                self.scaling_function = lambda trans_env: (trans_env - trans_env_mean_per_channel[:, : trans_env.shape[-1]]) / trans_env_std_per_channel[:, : trans_env.shape[-1]] 
                self.inv_scaling_function = lambda std_trans_env: std_trans_env * trans_env_std_per_channel[:, : std_trans_env.shape[-1]] + trans_env_mean_per_channel[:, : std_trans_env.shape[-1]]
    
    _trans_functions = ["log", "none"]
    _env_functions = ["hilbert", "moving_average", "moving_average_shifted", "first_order_lp", "constant_max", "constant_mean", "constant_one"]

    def _get_trans_function_by_name(self, name):
        if name == "log":
            return self._log_transform, self._inverse_log_transform
        elif name == "none":
            return lambda x: x, lambda x: x
        else:
            raise ValueError(f"Unknown transformation function: {name}. Supported functions are: {self._trans_functions}")
        
    def _get_env_function_by_name(self, name):
        if name == "hilbert":
            return self._hilbert_env
        elif name == "moving_average":
            return self._moving_average_env
        elif name == "moving_average_shifted":
            return self._moving_average_env_shifted
        elif name == "first_order_lp":
            return self._first_order_lp_env
        elif name == "constant_max":
            pass 
        elif name == "constant_mean":
            pass 
        elif name == "constant_one":
            pass 
        else:
            raise ValueError(f"Unknown envelope function: {name}. Supported functions are: {self._env_functions}")
        
    # ENVELOPES
    @staticmethod
    def _hilbert_env(signal):
        return np.abs(hilbert(signal))
    
    @staticmethod
    def _moving_average_env(signal, window_size=100):
        return np.apply_along_axis(lambda s: np.convolve(np.abs(s), np.ones(window_size)/window_size, mode='same'), axis=-1, arr=signal)
    
    @staticmethod
    def _moving_average_env_shifted(signal, window_size=100):
        moving_avg_envelopes = np.apply_along_axis(lambda s: np.convolve(np.abs(s), np.ones(window_size)/window_size, mode='same'), axis=-1, arr=signal)
        max_indices = np.argmax(signal, axis=-1)
        max_values = np.take_along_axis(signal, max_indices[..., np.newaxis], axis=-1)
        values = np.take_along_axis(moving_avg_envelopes, max_indices[..., np.newaxis], axis=-1)
        moving_avg_envelopes_shifted = moving_avg_envelopes + (max_values - values)
        return moving_avg_envelopes_shifted

    @staticmethod
    def _first_order_lp_env(signal, k_env=0.98):
        def first_order_lp_env_fun(s):
            envelope = np.zeros_like(s)
            s = np.abs(s)
            for i in range(1, len(s)):
                if s[i] > s[i-1] and s[i] > envelope[i-1]:
                    envelope[i] = s[i]
                else:
                    envelope[i] = k_env*envelope[i-1] + (1-k_env)*s[i] # first order discrete time low pass filter

            return envelope        
        return np.apply_along_axis(first_order_lp_env_fun, axis=-1, arr=signal)

    @staticmethod
    def _constant_max_env(signal):
        return np.max(signal, axis=-1)  
    
    @staticmethod
    def _constant_mean_env(signal):
        return np.mean(signal, axis=-1)
    
    @staticmethod
    def _constant_one_env(signal):
        return 1
    
    # TRANSFORMATION FUNCTIONS
    @staticmethod
    def _log_transform(x, log_offset=1e-5):
        return np.log10(x + log_offset)
    
    @staticmethod
    def _inverse_log_transform(x, log_offset=1e-5):
        return 10 ** x - log_offset
    
    ## --- Representation methods --- ##
    
    def _get_representation(self, signal):
        envelope = self.env_function(signal, **self.env_function_params)
        scaled_signal = np.divide(signal, envelope, out=np.zeros_like(signal), where=envelope!=0) # when envelope is 0, the signal is also 0. Hence, the scaled signal should also be 0.
        
        trans_envelope = self.trans_function(envelope, **self.env_transform_params) 
        scaled_envelope = self.scaling_function(trans_envelope)

        ch_axis = 0 if len(signal.shape) == 2 else 1

        return np.concatenate([scaled_envelope, scaled_signal], axis=ch_axis) # The model will learn to associated channels of the envelope with the corresponding channels of the signal

    def _invert_representation(self, representation):
        num_channels = representation.shape[1] // 2
        scaled_signal = representation[:, num_channels:, :]
        scaled_trans_envelope = representation[:, :num_channels, :]

        trans_envelope = self.inv_scaling_function(scaled_trans_envelope)
        
        signal = scaled_signal * self.inv_trans_function(trans_envelope, **self.env_transform_params)
        return signal
    
    def _plot(self, raw_waveform, title, inverted_waveform):
        n_channels = self.config.num_channels
        inverted_waveform = self.invert_representation(raw_waveform.reshape(1,-1))[0] if inverted_waveform is None else inverted_waveform
        fig, axs = plt.subplots(n_channels, 3, figsize=(15, 15))
        
        for c in range(n_channels):
            axs[c, 0].plot(raw_waveform[c, :], label=f'Channel {c}')
            axs[c, 1].plot(raw_waveform[n_channels+c, :], label=f'Channel {c}')
            axs[c, 2].plot(inverted_waveform[c, :], label=f'Channel {c}')   
            axs[c, 0].set_title('Gen. Transformed Envelope')
            axs[c, 1].set_title('Gen. Scaled Signal')
            axs[c, 2].set_title('Gen. Inverted Signal')
            axs[c, 0].legend(), axs[c, 1].legend(), axs[c, 2].legend()
        
        fig.suptitle(f'Cond. params: {title}')
        plt.show()    
    
    def _plot_representation(self, channel):
        signal = self.config.example_signal[channel, :]
        envelope = self.env_function(signal, **self.env_function_params)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(signal, alpha=0.6, label='signal')
        ax1.plot(envelope, linewidth=1,  label='envelope')
        title = f"Envelope function: {self.env_function.__name__}"
        ax1.set_title(title)
        ax1.legend()

        ax2 = fig.add_subplot(212)
        sig_scaled_by_envelope = np.nan_to_num(signal/envelope, nan=0., posinf=0., neginf=0.) # since if the envelope is 0, the signal is also small. 

        ax2.plot(sig_scaled_by_envelope)
        ax2.set_title("Signal scaled by its envelope")
        
        plt.tight_layout()
        plt.show()

    def _test(self, waveforms):
        for trans_fun in self._trans_functions:
            for env_fun in self._env_functions:
                repr_config = SignalWithEnvelope(env_fun, {}, trans_fun, {}, scaling={"type": "normalize", "scalar": True})
                repr_config.test_inversion(waveforms)
           

    def _update_stats(self, config: Config):
        # TODO
        # Compute the statistics of the transformed envelope (mean, std, min, max) over a subset of the training dataset
        # Save the statistics in a pickle file
        pass    
        

class LogSpectrogram(Representation):
    """Represents a signal as a log-spectrogram.

    Parameters
    ----------
    stft_channels : int, default=512
        Number of channels to use in the Short-Time Fourier Transform (STFT).
    hop_size : int, default=16
        Hop size in the STFT.
    clip : float, default=1e-8
        Clip value for spectrogram before taking the logarithm.
    log_max : float, default=3
        Empirical maximum value for the log-spectrogram. Used to normalize the log-spectrogram.
    output_shape : tuple, default=None
        Shape of the returned spectrogram.
        If given, the spectrogram will be padded or truncated to this shape.
    internal_shape : tuple, default=None
        Shape of the spectrogram passed to the inverse transform.
        If not given, it will be inferred during the first call to `get_representation`.
    library : str, default="librosa"
        Library to use for the STFT. Currently `librosa` and `tifresi` are supported.
    """

    def __init__(
        self,
        stft_channels=512,
        hop_size=16,
        clip=1e-8,
        log_max=3,
        output_shape=None,
        internal_shape=None,
        library="librosa",
        config=Config(),
    ):
        super().__init__(config)
        self.clip = clip
        self.log_clip = np.log(clip)
        self.log_max = log_max
        self.output_shape = output_shape
        self.internal_shape = internal_shape

        if library == "librosa":
            from librosa import griffinlim
            from librosa.core import stft

            self.stft = lambda x: stft(x, n_fft=stft_channels, hop_length=hop_size)
            self.istft = lambda x: griffinlim(
                x, hop_length=hop_size, n_fft=stft_channels, n_iter=128, random_state=0
            )

        elif library == "tifresi":
            from tifresi.stft import GaussTruncTF

            stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
            self.stft = stft_system.spectrogram
            self.istft = stft_system.invert_spectrogram
    

    def adjust_to_shape(self, x, shape=None):
        if not shape:
            return x
        if x.shape[1] < shape[0]:
            x = np.pad(x, ((0, 0), (0, shape[0] - x.shape[1]), (0, 0)), mode="constant")
        elif x.shape[1] > shape[0]:
            x = x[:, :shape[0], :]
        if x.shape[2] < shape[1]:
            x = np.pad(x, ((0, 0), (0, 0), (0, shape[1] - x.shape[2])), mode="constant")
        elif x.shape[2] > shape[1]:
            x = x[:, :, :shape[1]]

        return x

    def get_spectrogram(self, signal):
        shape = signal.shape
        signal = signal.reshape(-1, shape[-1])  # flatten trailing dimensions
        spec = np.array([self.stft(x) for x in signal])
        if not self.internal_shape:
            self.internal_shape = spec.shape[1:]
        spec = self.adjust_to_shape(spec, self.output_shape)
        spec = spec.reshape(shape[:-1] + spec.shape[1:])  # restore trailing dimensions
        return spec

    def invert_spectrogram(self, spec):
        shape = spec.shape
        spec = spec.reshape(-1, shape[-2], shape[-1])  # flatten trailing dimensions
        spec = self.adjust_to_shape(spec, self.internal_shape)
        signal = np.array([self.istft(x) for x in spec])
        signal = signal.reshape(shape[:-2] + signal.shape[1:])  # restore trailing dimensions
        return signal

    def _get_representation(self, signal):
        spec = self.get_spectrogram(signal)
        spec = np.abs(spec)
        log_spec = np.log(np.clip(spec, self.clip, None))  # [log_clip, log_max] # TODO: ask, why log?
        norm_log_spec = (log_spec - self.log_clip) / (self.log_max - self.log_clip)  # [0, 1]
        norm_log_spec = norm_log_spec * 2 - 1  # [-1, 1]
        return norm_log_spec

    def _invert_representation(self, representation):
        norm_log_spec = (representation + 1) / 2
        log_spec = norm_log_spec * (self.log_max - self.log_clip) + self.log_clip
        spec = np.exp(log_spec)
        return self.invert_spectrogram(spec)[..., : self.config.signal_length] # TODO: check why when calling _invert_representation alone, the signal_length is not 5472
    
    def _plot(self, raw_waveform, title, inverted_waveform, channel=None):
        channels = [c for c in range(self.config.num_channels)] if channel is None else [channel] 
        inverted_waveform = self.invert_representation(raw_waveform[None, ...])[0] if inverted_waveform is None else inverted_waveform
        fig, axs = plt.subplots(len(channels), 2, figsize=(15, 15))
        axs = np.atleast_2d(axs)
        time_ax = np.arange(0, self.config.signal_length) / self.config.fs
        for c in channels:
            axs[c, 0].imshow(raw_waveform[c, :])
            axs[c, 0].text(0.95, 0.95, f'Channel {c}', ha='right', va='bottom', transform=axs[c, 0].transAxes, bbox={'facecolor': 'white', 'pad': 5})
            axs[c, 0].set_xlabel('Time Bins'), axs[c, 0].set_ylabel('Freq. Bins')
            axs[c, 0].set_title('Gen. Normalized Log-Spectrogram (STFT)')
            axs[c, 1].plot(time_ax, inverted_waveform[c, :self.config.signal_length], label=f'Channel {c}')   
            axs[c, 1].set_xlabel('Time (s)')
            axs[c, 1].set_title('Gen. Inverted Signal')
            axs[c, 1].legend()
        
        if title:
            fig.suptitle(f'Cond. params: {title}')
        plt.tight_layout()
        plt.show()   


    def _plot_representation(self, channel):
        signal = self.config.example_signal[channel, :]
        spectrogram = self._get_representation(signal)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(signal)
        ax1.set_title(f"Signal - channel {channel}")

        ax2 = fig.add_subplot(212) 
        ax2.imshow(spectrogram)
        ax2.set_title(f"Spectrogram (STFT)- channel {channel}")
        ax2.set_xlabel('Time Bins'), ax2.set_ylabel('Freq. Bins')
        
        plt.tight_layout()
        plt.show()

    def _test(self, waveforms):
        return LogSpectrogram()._test_inversion(waveforms)    
        
    
    def _update_stats(self, config: Config):
        pass 
     
