import numpy as np
import torch

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from matplotlib import pyplot as plt
from scipy.signal import hilbert
from librosa.util import fix_length

from tqdne.conf import Config


@staticmethod
def to_numpy(x):
    if isinstance(x, Sequence):
        return x.__class__(to_numpy(v) for v in x)
    elif isinstance(x, Mapping):
        return x.__class__((k, to_numpy(v)) for k, v in x.items())
    else:
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x
    
@staticmethod
def to_torch(x, device='cuda', dtype=torch.float32):
    if isinstance(x, Sequence):
        return x.__class__(to_torch(v, device) for v in x)
    elif isinstance(x, Mapping):
        return x.__class__((k, to_torch(v, device)) for k, v in x.items())
    else:
        return torch.from_numpy(x).to(torch.device(device), dtype=dtype) if isinstance(x, np.ndarray) else x.to(torch.device(device), dtype=dtype)


class Representation(ABC):

    def __init__(self, config=Config()):
        self.config = config

    def __str__(self):
        return f"{self.__class__.__name__} - {self.parameters}"

    def get_representation(self, signal):
        return self._get_representation(signal)

    @abstractmethod
    def _get_representation(self, signal):
        pass

    def invert_representation(self, representation):
        return self._invert_representation(representation)

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
    def _plot_representation(self, signal=None, channel=0):
        pass

    def plot_representation(self, downsampling_factor, channel=0): 
        from tqdne.dataset import downsample_waveform
        signal = downsample_waveform(self.config.example_signal, downsampling_factor)
        return self._plot_representation(signal, channel)   
    
    @abstractmethod
    def _plot_distribution(self, pred_raw_waveforms, test_raw_waveforms):
        pass

    def plot_distribution(self, pred_raw_waveforms, test_raw_waveforms):
        return self._plot_distribution(pred_raw_waveforms, test_raw_waveforms)

    @abstractmethod
    def _test(self, waveforms):
        pass
    
    @staticmethod
    def test(repr, waveforms):
        repr._test(repr, waveforms)

    def _test_inversion(self, waveforms):
        assert np.allclose(waveforms, self.invert_representation(self.get_representation(waveforms)), atol=1e-6) 

    @abstractmethod
    def _get_name(self, FLAGS, name=None):
        pass 

    def get_name(self, FLAGS, name=None):
        return self._get_name(FLAGS, name)   
    
    def _get_num_channels(self, original_num_channels: int) -> int:
        return original_num_channels

    def get_num_channels(self, original_num_channels: int) -> int:
        return self._get_num_channels(original_num_channels)


class Signal(Representation):
    def __init__(self, scaling=None, config=Config()):
        super().__init__(config)

        self.scaling = scaling
        if scaling is not None:
            if self.scaling["type"] == "normalize":
                if self.scaling["scalar"]:
                    # Signal statistics for normalization
                    dataset_stats_dict = config.signal_statistics 
                    max_per_channel = np.array([dataset_stats_dict[ch]['max'] for ch in dataset_stats_dict.keys()])[:, np.newaxis]
                    min_per_channel = np.array([dataset_stats_dict[ch]['min'] for ch in dataset_stats_dict.keys()])[:, np.newaxis]
                    #Normalize the transformed envelope to the range [-1, 1]
                    self.scaling_function = lambda trans_env: 2 * (trans_env - min_per_channel) / (max_per_channel - min_per_channel) - 1
                    self.inv_scaling_function = lambda norm_trans_env: (norm_trans_env + 1) * (max_per_channel - min_per_channel) / 2 + min_per_channel
                else:
                    raise NotImplementedError("Channel-wise normalization is not implemented yet -- missing statistics")
            elif self.scaling["type"] == "standardize":
                if self.scaling["scalar"]:
                    raise NotImplementedError("Scalar standardization is not implemented yet")
                else:
                    raise NotImplementedError("Channel-wise standardization is not implemented yet -- missing statistics")
            elif self.scaling["type"] == "none":
                self.scaling_function = lambda x: x
                self.inv_scaling_function = lambda x: x
            else:
                raise ValueError(f"Unknown scaling function: {self.scaling['type']}. Supported functions are: ['normalize', 'standardize', 'none']")  
            
    def _get_name(self, FLAGS, name=None):
        if name:
            return name + f"_{self.__class__.__name__}"
        return f"{FLAGS.config.name}-pred:{FLAGS.config.model.scheduler_params.prediction_type}-{FLAGS.config.model.net_params.dims}D-downsampling:{FLAGS.downsampling_factor}_{FLAGS.config.data_repr.name}-{FLAGS.config.data_repr.params.scaling.type}-scalar:{FLAGS.config.data_repr.params.scaling.scalar}".replace(" ", "").replace("\n", "")

    def _get_representation(self, signal):
        signal = to_numpy(signal)
        if self.scaling is not None:
            return self.scaling_function(signal)
        return signal

    def _invert_representation(self, representation):
        representation = to_numpy(representation)
        if self.scaling is not None:
            return self.inv_scaling_function(representation)
        return representation

    def _plot(self, raw_waveform, title, inverted_waveform):
        fig, axs = plt.subplots(self.config.num_channels, 1, figsize=(15, 15))
        for c in range(self.config.num_channels):
            axs[c, 0].plot(raw_waveform[c, :], label=f'Channel {c}')
            axs[c, 0].set_title('Gen. Signal')
            axs[c, 0].legend()

        fig.suptitle(f'Cond. params: {title}')
        plt.tight_layout()
        plt.show()    

    def _plot_representation(self, signal, channel):
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
        time_ax = np.arange(0, signal.shape[-1]) / self.config.fs
        ax.plot(time_ax, signal[channel])
        ax.set_xlabel('Time [s]')
        ax.set_title(f"Signal - channel {channel}")
        plt.tight_layout()
        plt.show()

    def _plot_distribution(self, pred_raw_waveforms, test_raw_waveforms):
        mean_raw_pred = np.mean(pred_raw_waveforms, axis=0)
        std_raw_pred = np.std(pred_raw_waveforms, axis=0)
        mean_raw_test = np.mean(test_raw_waveforms, axis=0)
        std_raw_test = np.std(test_raw_waveforms, axis=0)

        time_ax = np.arange(0, self.config.signal_length) / self.config.fs
        fig, axs = plt.subplots(self.config.num_channels, 1, figsize=(10, 10))
        for i in range(self.config.num_channels):
            axs[i].plot(time_ax, mean_raw_test[i], label='Test Mean')
            axs[i].fill_between(time_ax, mean_raw_test[i] - std_raw_test[i], mean_raw_test[i] + std_raw_test[i], alpha=0.3, label=f'Test Std - num_samples={len(test_raw_waveforms)}')
            axs[i].plot(time_ax, mean_raw_pred[i], label='Generated Mean')
            axs[i].fill_between(time_ax, mean_raw_pred[i] - std_raw_pred[i], mean_raw_pred[i] + std_raw_pred[i], alpha=0.3, label=f'Generated Std - num_samples={len(pred_raw_waveforms)}')
            axs[i].set_title(f'Channel {i}')
            axs[i].legend()

        fig.tight_layout()
        plt.show()

    def _test(self, waveforms):
        # TODO: Maybe test scaling
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
                if "dataset_stats_file" in self.scaling:
                    import pickle
                    with open(self.scaling["dataset_stats_file"], 'rb') as pickle_file:
                        envelope_stats_dict = pickle.load(pickle_file)
                        max_trans_env_peak_per_channel = np.array([envelope_stats_dict[ch]['max'] for ch in envelope_stats_dict.keys()])[:, np.newaxis]
                        min_trans_env_peak_per_channel = np.array([envelope_stats_dict[ch]['min'] for ch in envelope_stats_dict.keys()])[:, np.newaxis]
                else:
                    dataset_stats_dict = config.signal_statistics 
                    # Assuming that maximum of the envelope coincides with the maximum of the signal and that the minimum of the envelope is 0.
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
            raise NotImplementedError()
            # if self.scaling["scalar"]:
            #     env_stats_per_channel = [dataset_stats_dict[ch] for ch in dataset_stats_dict.keys()]
            #     trans_env_mean_per_channel = np.array([env_stats['mean'] for env_stats in env_stats_per_channel]).reshape(-1, 1) # shape: (num_channels, 1)
            #     trans_env_std_per_channel = np.array([env_stats['std_dev'] for env_stats in env_stats_per_channel]).reshape(-1, 1) # shape: (num_channels, 1)
            #     self.scaling_function = lambda trans_env: (trans_env - trans_env_mean_per_channel) / trans_env_std_per_channel 
            #     self.inv_scaling_function = lambda std_trans_env: std_trans_env * trans_env_std_per_channel + trans_env_mean_per_channel
                
            # else:
            #     # Statistics of the transformed envelope (computer over a subset of the training dataset)
            #     env_stats_per_channel = [dataset_stats_dict[ch] for ch in dataset_stats_dict.keys()]
            #     trans_env_mean_per_channel = np.array([env_stats['mean_signal'] for env_stats in env_stats_per_channel]) # shape: (num_channels, signal_length)
            #     trans_env_std_per_channel = np.array([env_stats['std_dev_signal'] for env_stats in env_stats_per_channel]) # shape: (num_channels, signal_length)
            #     # Standardize the transformed envelope 
            #     self.scaling_function = lambda trans_env: (trans_env - trans_env_mean_per_channel[:, : trans_env.shape[-1]]) / trans_env_std_per_channel[:, : trans_env.shape[-1]] 
            #     self.inv_scaling_function = lambda std_trans_env: std_trans_env * trans_env_std_per_channel[:, : std_trans_env.shape[-1]] + trans_env_mean_per_channel[:, : std_trans_env.shape[-1]]

        elif self.scaling["type"] == "none":
            self.scaling_function = lambda x: x
            self.inv_scaling_function = lambda x: x
        
        else:
            raise ValueError(f"Unknown scaling function: {self.scaling['type']}. Supported functions are: ['normalize', 'standardize', 'none']") 

    def _get_name(self, FLAGS, name=None):
        if FLAGS.config.data_repr.params.scaling.type == "none":
            name = f"{FLAGS.config.name}-pred:{FLAGS.config.model.scheduler_params.prediction_type}-{FLAGS.config.model.net_params.dims}D-downsampling:{FLAGS.downsampling_factor}_{FLAGS.config.data_repr.name}-{FLAGS.config.data_repr.params.env_function}-{FLAGS.config.data_repr.params.env_function_params}-{FLAGS.config.data_repr.params.env_transform}-{FLAGS.config.data_repr.params.env_transform_params}".replace(" ", "").replace("\n", "")
        else:
            name = f"{FLAGS.config.name}-pred:{FLAGS.config.model.scheduler_params.prediction_type}-{FLAGS.config.model.net_params.dims}D-downsampling:{FLAGS.downsampling_factor}_{FLAGS.config.data_repr.name}-{FLAGS.config.data_repr.params.env_function}-{FLAGS.config.data_repr.params.env_function_params}-{FLAGS.config.data_repr.params.env_transform}-{FLAGS.config.data_repr.params.env_transform_params}-{FLAGS.config.data_repr.params.scaling.type}-scalar:{FLAGS.config.data_repr.params.scaling.scalar}".replace(" ", "").replace("\n", "")  
        return name             
    
    def _get_num_channels(self, original_num_channels):
        return 2 * original_num_channels

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
        # TODO: maybe the only one that makes sense is max, as the scaled signal in this case would be scaled to the max of the signal, while for the others it would increase the amplitude of the signal (i.e, not scaled)
        elif name == "constant_max":
            return self._constant_max_env 
        elif name == "constant_mean":
            return self._constant_mean_env 
        elif name == "constant_one":
            return self._constant_one_env 
        else:
            raise ValueError(f"Unknown envelope function: {name}. Supported functions are: {self._env_functions}")
        
    # ENVELOPES
    @staticmethod
    def _hilbert_env(signal):
        return np.abs(hilbert(signal))
    
    @staticmethod
    def _moving_average_env(signal, window_size=100, scale=1):
        return scale * np.apply_along_axis(lambda s: np.convolve(np.abs(s), np.ones(window_size)/window_size, mode='same'), axis=-1, arr=signal)
 
    
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
        return np.ones_like(signal) * np.max(np.abs(signal), axis=-1, keepdims=True)
    
    @staticmethod
    def _constant_mean_env(signal):
        return np.ones_like(signal) * np.mean(np.abs(signal), axis=-1, keepdims=True)
    
    @staticmethod
    def _constant_one_env(signal):
        return np.ones_like(signal)
    
    # TRANSFORMATION FUNCTIONS
    @staticmethod
    def _log_transform(x, log_offset=1e-5):
        return np.log10(x + log_offset)
    
    @staticmethod
    def _inverse_log_transform(x, log_offset=1e-5):
        return 10 ** x - log_offset
    
    ## --- Representation methods --- ##
    
    def _get_representation(self, signal):
        signal = to_numpy(signal)

        envelope = self.env_function(signal, **self.env_function_params)
        scaled_signal = np.divide(signal, envelope, out=np.zeros_like(signal), where=envelope!=0) # when envelope is 0, the signal is also 0. Hence, the scaled signal should also be 0.
        
        trans_envelope = self.trans_function(envelope, **self.env_transform_params) 
        scaled_envelope = self.scaling_function(trans_envelope)

        ch_axis = 0 if len(signal.shape) == 2 else 1

        return np.concatenate([scaled_envelope, scaled_signal], axis=ch_axis) # The model will learn to associated channels of the envelope with the corresponding channels of the signal

    def _invert_representation(self, representation):
        representation = to_numpy(representation)

        num_channels = representation.shape[1] // 2
        scaled_signal = representation[:, num_channels:, :]
        scaled_trans_envelope = representation[:, :num_channels, :]

        trans_envelope = self.inv_scaling_function(scaled_trans_envelope)
        
        signal = scaled_signal * self.inv_trans_function(trans_envelope, **self.env_transform_params)
        return signal
    
    ## --- Plotting methods --- ##
    def _plot(self, raw_waveform, title, inverted_waveform):
        n_channels = self.config.num_channels
        time_ax = np.arange(0, self.config.signal_length) / self.config.fs
        fig, axs = plt.subplots(n_channels, 3, figsize=(20, 15))
        for c in range(n_channels):
            axs[c, 0].plot(time_ax, raw_waveform[c, :], label=f'Channel {c}')
            axs[c, 1].plot(time_ax, raw_waveform[n_channels+c, :], label=f'Channel {c}')
            axs[c, 2].plot(time_ax, inverted_waveform[c, :], label=f'Channel {c}')   
            axs[c, 0].set_title('Gen. Transformed Envelope')
            axs[c, 1].set_title('Gen. Scaled Signal')
            axs[c, 2].set_title('Gen. Inverted Signal')
            axs[c, 2].set_ylabel('$m/s^2$')
            axs[c, 0].legend(), axs[c, 1].legend(), axs[c, 2].legend()
        
        axs[c, 0].set_xlabel('Time [s]'), axs[c, 1].set_xlabel('Time [s]'), axs[c, 2].set_xlabel('Time [s]')
        fig.suptitle(f'Cond. params: {title}')
    
        plt.tight_layout()
        plt.show()    
    
    def _plot_representation(self, signal, channel):

        envelope = self.env_function(signal, **self.env_function_params)

        time_ax = np.arange(0, signal.shape[-1]) / self.config.fs

        fig = plt.figure(figsize=(15, 9))
        ax1 = fig.add_subplot(311)
        ax1.plot(time_ax, signal[channel], alpha=0.6, label='Signal')
        ax1.plot(time_ax, envelope[channel], linewidth=1,  label='Envelope')
        title = f"Envelope function: {self.env_function.__name__} - {self.env_function_params}" if self.env_function_params else f"Envelope function: {self.env_function.__name__}"
        ax1.set_title(title)
        ax1.set_xlabel('Time [s]')
        ax1.legend()

        repr = self._get_representation(signal)
        num_channels = self.config.num_channels

        ax2 = fig.add_subplot(312)
        sig_scaled_by_envelope = repr[num_channels:, :]
        ax2.plot(time_ax, sig_scaled_by_envelope[channel])
        ax2.set_title("Signal scaled by the envelope")
        ax2.set_xlabel('Time [s]')  

        ax3 = fig.add_subplot(313)
        scaled_envelope = repr[:num_channels, :]
        ax3.plot(time_ax, scaled_envelope[channel])
        ax3.set_title("Transformed and scaled envelope")
        ax3.set_xlabel("Time [s]")
        
        plt.tight_layout()
        plt.show()

    def _plot_distribution(self, pred_raw_waveforms, test_raw_waveforms):
        mean_raw_pred = np.mean(pred_raw_waveforms, axis=0)
        std_raw_pred = np.std(pred_raw_waveforms, axis=0)
        mean_raw_test = np.mean(test_raw_waveforms, axis=0)
        std_raw_test = np.std(test_raw_waveforms, axis=0)

        time_ax = np.arange(0, self.config.signal_length) / self.config.fs
        fig, axs = plt.subplots(self.config.num_channels, 2, figsize=(10, 10))
        for i in range(self.config.num_channels):
            axs[i, 0].plot(time_ax, mean_raw_test[i], label='Test Mean')
            axs[i, 0].fill_between(time_ax, mean_raw_test[i] - std_raw_test[i], mean_raw_test[i] + std_raw_test[i], alpha=0.3, label=f'Test Std -- num_samples={len(test_raw_waveforms)}')
            axs[i, 0].plot(time_ax, mean_raw_pred[i], label='Generated Mean')
            axs[i, 0].fill_between(time_ax, mean_raw_pred[i] - std_raw_pred[i], mean_raw_pred[i] + std_raw_pred[i], alpha=0.3, label=f'Generated Std -- num_samples={len(pred_raw_waveforms)}')
            axs[i, 0].set_title(f'Transformed Log Envelope - Channel {i}')
            axs[i, 0].legend()
            axs[i, 1].plot(time_ax, mean_raw_test[i+self.config.num_channels], label='Test Mean')
            axs[i, 1].fill_between(time_ax, mean_raw_test[i+self.config.num_channels] - std_raw_test[i+self.config.num_channels], mean_raw_test[i+self.config.num_channels] + std_raw_test[i+self.config.num_channels], alpha=0.3, label=f'Test Std -- num_samples={len(test_raw_waveforms)}')
            axs[i, 1].plot(time_ax, mean_raw_pred[i+self.config.num_channels], label='Generated Mean')
            axs[i, 1].fill_between(time_ax, mean_raw_pred[i+self.config.num_channels] - std_raw_pred[i+self.config.num_channels], mean_raw_pred[i+self.config.num_channels] + std_raw_pred[i+self.config.num_channels], alpha=0.3, label=f'Generated Std -- num_samples={len(pred_raw_waveforms)}')
            axs[i, 1].set_title(f'Scaled Signal - Channel {i}')
            axs[i, 1].legend()

        fig.tight_layout()
        plt.show()    

    ## --- Testing methods --- ##
    def _test(self, waveforms):
        for trans_fun in self._trans_functions:
            for env_fun in self._env_functions:
                repr_config = SignalWithEnvelope(env_fun, {}, trans_fun, {}, scaling={"type": "normalize", "scalar": True})
                repr_config.test_inversion(waveforms)
            
        

class LogSpectrogram(Representation):
    def __init__(
        self,
        output_signal_length,
        stft_channels=512,
        hop_size=None,
        griffin_lim_iterations=128,
        clip=1e-8,
        log_max=3,
        library="librosa",
        device="cpu",
        config=Config(),
    ):
        """
        LogSpectrogram representation class.

        Args:
            output_signal_length (int): Length of the output signal.
            stft_channels (int): Number of channels in the STFT.
            hop_size (int): Hop size for the STFT. If None, it is set to stft_channels // 4.
            griffin_lim_iterations (int): Number of iterations for the Griffin-Lim algorithm. Default is 128.
            clip (float): Clip value for the logarithm operation.
            log_max (float): Maximum value for the logarithm operation.
            library (str): Library to use for the STFT and inverse STFT operations. Options are "librosa", "tifresi", and "torch".
            device (str): Device to use for the STFT and inverse STFT operations. Options are "cpu" and "cuda".
            config (Config): Configuration object.

        """
        super().__init__(config)
        self.clip = clip
        self.log_clip = np.log(clip)
        self.log_max = log_max
        self.stft_channels = stft_channels
        self.hop_size = hop_size if hop_size is not None else stft_channels // 4
        self.griffin_lim_iterations = griffin_lim_iterations
        self.library = library

        self.output_signal_length = output_signal_length
        self.internal_signal_length = output_signal_length + self.stft_channels // 2
        self._set_stft_functions(self.internal_signal_length)

        self.device = device
        if library == 'librosa' and device != 'cpu': 
            print(f"Warning: librosa is not CUDA compatible. The device will be set to 'cpu'.")
            self.device = 'cpu'
        if device != 'cpu':
            if torch.device('cuda' if torch.cuda.is_available() else 'cpu').type == 'cpu':
                print(f"Warning: CUDA is not available. The device will be set to 'cpu'.")
                self.device = 'cpu'
        self.to_array_fun = to_numpy if self.device == 'cpu' else to_torch

        #self._set_stft_functions()

    def _set_stft_functions(self, signal_length=None):
        """
        Set the STFT and inverse STFT functions based on the selected library.

        Args:
            signal_length (int): Length of the signal used for the inverse STFT.

        """
        if self.library == "librosa":
            from librosa import griffinlim, stft

            self.stft = lambda x: stft(x, n_fft=self.stft_channels, hop_length=self.hop_size)
            self.istft = lambda x: griffinlim(
                x, hop_length=self.hop_size, n_fft=self.stft_channels, length=signal_length, n_iter=self.griffin_lim_iterations, random_state=0
            )

        elif self.library == "tifresi":
            from tifresi.stft import GaussTruncTF

            print("Warning: Tifresi is not yet fully supported")
            stft_system = GaussTruncTF(hop_size=self.hop_size, stft_channels=self.stft_channels)
            self.stft = stft_system.spectrogram
            self.istft = stft_system.invert_spectrogram

        elif self.library == "torch":
            #window = torch.hann_window(self.stft_channels, device=torch.device(self.device))
            #window = torch.hann_window(self.stft_channels)#.to(self.device)
            window = torch.hann_window(self.stft_channels)
            self.stft = lambda x: torch.stft(
                to_torch(x, device=self.device),
                n_fft=self.stft_channels, 
                hop_length=self.hop_size, 
                window=to_torch(window, device=self.device), 
                return_complex=True
                ).numpy()
            from torchaudio.functional import griffinlim
            self.istft = lambda x: griffinlim(
                to_torch(x, device=self.device), 
                n_fft=self.stft_channels, 
                hop_length=self.hop_size, 
                length=signal_length,  
                window=to_torch(window, device=self.device), 
                win_length=self.stft_channels,
                power=1, 
                n_iter=self.griffin_lim_iterations,
                momentum=.99, 
                rand_init=False
                ).numpy()        

    def _get_name(self, FLAGS, name=None):
        """
        Get the name of the representation.

        Args:
            FLAGS: Flags object.
            name (str): Name to append to the representation name.

        Returns:
            str: Representation name.

        """
        if name:
            return name + f"_{self.__class__.__name__}-stft_ch:{self.stft_channels}-hop_size:{self.hop_size}"
        return f"{FLAGS.config.name}-pred:{FLAGS.config.model.scheduler_params.prediction_type}-{FLAGS.config.model.net_params.dims}D-downsampling:{FLAGS.downsampling_factor}_{FLAGS.config.data_repr.name}-stft_ch:{self.stft_channels}-hop_size:{self.hop_size}".replace(" ", "").replace("\n", "")

    def _get_spectrogram(self, signal):
        """
        Get the spectrogram of the input signal.

        Args:
            signal (ndarray): Input signal.

        Returns:
            ndarray: Spectrogram.

        """
        signal_pad = fix_length(to_numpy(signal), size=self.internal_signal_length)
        shape = signal_pad.shape
        signal_pad = signal_pad.reshape(-1, shape[-1])  # flatten trailing dimensions
        signal_pad = self.to_array_fun(signal_pad)
        spec = np.array([self.stft(x) for x in signal_pad])
        spec = spec.reshape(shape[:-1] + spec.shape[1:])  # restore trailing dimensions
        return spec

    def _invert_spectrogram(self, spec):
        """
        Invert the spectrogram to obtain the signal.

        Args:
            spec (ndarray): Spectrogram.

        Returns:
            ndarray: Inverted signal.

        """
        shape = spec.shape
        spec = spec.reshape(-1, shape[-2], shape[-1])  # flatten trailing dimensions
        spec = self.to_array_fun(spec)
        signal = np.array([self.istft(x) for x in spec])
        signal = signal.reshape(shape[:-2] + signal.shape[1:])  # restore trailing dimensions
        signal = signal[..., :self.output_signal_length]  # remove padding
        return signal

    def _get_representation(self, signal):
        """
        Get the representation of the input signal.

        Args:
            signal (ndarray): Input signal.

        Returns:
            ndarray: Representation.

        """
        spec = self._get_spectrogram(signal)
        spec = np.abs(spec)
        log_spec = np.log(np.clip(spec, self.clip, None))  # [log_clip, log_max]
        norm_log_spec = (log_spec - self.log_clip) / (self.log_max - self.log_clip)  # [0, 1]
        norm_log_spec = norm_log_spec * 2 - 1  # [-1, 1]
        return norm_log_spec

    def _invert_representation(self, representation):
        """
        Invert the representation to obtain the signal.

        Args:
            representation (ndarray): Representation.

        Returns:
            ndarray: Inverted signal.

        """
        norm_log_spec = (representation + 1) / 2
        log_spec = norm_log_spec * (self.log_max - self.log_clip) + self.log_clip
        try:
            spec = torch.exp(log_spec)
        except TypeError:
            spec = np.exp(log_spec)
        return self._invert_spectrogram(spec)
    
    ## --- Plotting methods --- ##
    
    def _plot(self, raw_waveform, title, inverted_waveform, channel=None):
        """
        Plot the raw waveform and the inverted waveform.

        Args:
            raw_waveform (ndarray): Raw waveform.
            title (str): Title for the plot.
            inverted_waveform (ndarray): Inverted waveform.
            channel (int): Channel to plot. If None, plot all channels.

        """
        channels = [c for c in range(self.config.num_channels)] if channel is None else [channel] 
        fig, axs = plt.subplots(len(channels), 2, figsize=(15, 15))
        axs = np.atleast_2d(axs)
        time_ax = np.arange(0, self.config.signal_length) / self.config.fs
        for c in channels:
            axs[c, 0].imshow(raw_waveform[c, :])
            axs[c, 0].text(0.95, 0.95, f'Channel {c}', ha='right', va='bottom', transform=axs[c, 0].transAxes, bbox={'facecolor': 'white', 'pad': 5})
            axs[c, 0].set_xlabel('Time Bins'), axs[c, 0].set_ylabel('Freq. Bins')
            axs[c, 0].set_title('Gen. Normalized Log-Spectrogram (STFT)')
            axs[c, 1].plot(time_ax, inverted_waveform[c, :self.config.signal_length], label=f'Channel {c}')   
            axs[c, 1].set_xlabel('Time [s]')
            axs[c, 1].set_ylabel('$[m/s^2]$')
            axs[c, 1].set_title('Gen. Inverted Signal')
            axs[c, 1].legend()
        
        if title:
            fig.suptitle(f'Cond. params: {title}')
        plt.tight_layout()
        plt.show()   

    def _plot_representation(self, signal, channel):
        """
        Plots the signal and its spectrogram representation.

        Args:
            signal (ndarray): The input signal.
            channel (int): The channel number.

        Returns:
            None
        """

        time_ax = np.arange(0, signal.shape[-1]) / self.config.fs
        spectrogram = self._get_representation(signal)

        fig = plt.figure(figsize=(15, 9))
        ax1 = fig.add_subplot(211)
        ax1.plot(time_ax, signal[channel])
        ax1.set_title(f"Signal - channel {channel}")
        ax1.set_xlabel('Time [s]')

        ax2 = fig.add_subplot(212)
        ax2.imshow(spectrogram[channel])
        ax2.set_title(f"Spectrogram (STFT) - channel {channel}")
        ax2.set_xlabel('Time Bins'), ax2.set_ylabel('Freq. Bins')
        ax2.set_aspect('auto')  # Set same width as ax1

        plt.tight_layout()
        plt.show()


    def _plot_distribution(self, pred_raw_waveforms, test_raw_waveforms):
        """
        Plot the distribution of the predicted and test raw waveforms.

        Args:
            pred_raw_waveforms (ndarray): Predicted raw waveforms.
            test_raw_waveforms (ndarray): Test raw waveforms.

        """
        mean_raw_pred = np.mean(pred_raw_waveforms, axis=0)
        std_raw_pred = np.std(pred_raw_waveforms, axis=0)
        mean_raw_test = np.mean(test_raw_waveforms, axis=0)
        std_raw_test = np.std(test_raw_waveforms, axis=0)

        fig, axs = plt.subplots(self.config.num_channels, 2, figsize=(10, 10))
        for i in range(self.config.num_channels):
            diff_abs = axs[i, 0].imshow(np.abs(mean_raw_test[i] - mean_raw_pred[i]), cmap='hot', interpolation='nearest')
            axs[i, 0].set_xlabel('Time Bins'), axs[i, 0].set_ylabel('Freq. Bins')
            axs[i, 0].set_title(f'Mean Abs. Diff. (pred. vs test) - Channel {i}')
            fig.colorbar(diff_abs, ax=axs[i, 0])
            diff_std = axs[i, 1].imshow(np.abs(std_raw_test[i] - std_raw_pred[i]), cmap='hot', interpolation='nearest')
            axs[i, 1].set_title(f'Std Abs. Diff. (pred. vs test) - Channel {i}')
            axs[i, 1].set_xlabel('Time Bins'), axs[i, 1].set_ylabel('Freq. Bins')
            fig.colorbar(diff_std, ax=axs[i, 1])
        fig.tight_layout()
        plt.show()    
                 

    ## --- Testing methods --- ##    

    def _test(self, waveforms):
        """
        Test the inversion of the waveforms.

        Args:
            waveforms (ndarray): Waveforms to test.

        Returns:
            bool: True if the inversion is successful, False otherwise.

        """
        return LogSpectrogram()._test_inversion(waveforms)    
    