"""Utilities for profiling FLOPs and performance metrics."""

import time
from contextlib import contextmanager
from typing import Any

import torch as th


class FLOPsCounter:
    """
    Context manager for counting FLOPs during model execution.

    Usage:
        counter = FLOPsCounter()
        with counter.profile():
            output = model(input)
        print(f"GFLOPs: {counter.gflops:.2f}")
    """

    def __init__(self):
        self.total_flops = 0
        self.total_time = 0.0
        self.n_calls = 0

    @contextmanager
    def profile(self):
        """Profile a code block."""
        # Try using torch profiler if available
        try:
            with th.profiler.profile(
                activities=[
                    th.profiler.ProfilerActivity.CPU,
                    th.profiler.ProfilerActivity.CUDA,
                ],
                with_flops=True,
            ) as prof:
                start_time = time.perf_counter()
                yield
                end_time = time.perf_counter()

            # Extract FLOPs from profiler
            self.total_time += end_time - start_time
            self.n_calls += 1

            # Sum up FLOPs from all events
            flops = sum([event.flops for event in prof.key_averages() if event.flops > 0])
            self.total_flops += flops

        except Exception as e:
            print(f"Warning: FLOPs profiling failed: {e}")
            print("Falling back to timing only...")
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()
            self.total_time += end_time - start_time
            self.n_calls += 1

    @property
    def gflops(self) -> float:
        """Total GFLOPs executed."""
        return self.total_flops / 1e9

    @property
    def avg_time(self) -> float:
        """Average time per call in seconds."""
        return self.total_time / max(self.n_calls, 1)

    @property
    def gflops_per_second(self) -> float:
        """GFLOPs per second throughput."""
        if self.total_time > 0:
            return self.gflops / self.total_time
        return 0.0

    def print_summary(self, title: str = "Profiling Summary"):
        """Print a summary of profiling results."""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Total calls:        {self.n_calls}")
        print(f"Total time:         {self.total_time:.4f} s")
        print(f"Average time:       {self.avg_time*1000:.2f} ms")
        print(f"Total GFLOPs:       {self.gflops:.2f}")
        print(f"Throughput:         {self.gflops_per_second:.2f} GFLOPS/s")
        print(f"{'='*60}\n")


def profile_model_static(model: th.nn.Module, input_shape: tuple, device: str = "cuda") -> dict[str, Any]:
    """
    Profile a model statically to estimate FLOPs.

    This uses fvcore if available, otherwise falls back to a simpler method.

    Parameters
    ----------
    model : torch.nn.Module
        The model to profile
    input_shape : tuple
        Shape of input tensor (batch_size, channels, ...)
    device : str
        Device to run on (default: 'cuda')

    Returns
    -------
    dict
        Dictionary with profiling results including 'gflops', 'params'
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count

        model.eval()
        dummy_input = th.randn(input_shape).to(device)

        with th.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            params = parameter_count(model)

        total_flops = flops.total()
        total_params = params[""]

        return {
            "gflops": total_flops / 1e9,
            "params": total_params,
            "params_m": total_params / 1e6,
        }

    except ImportError:
        print("Warning: fvcore not installed. Install with: pip install fvcore")
        print("Falling back to parameter counting only...")

        total_params = sum(p.numel() for p in model.parameters())
        return {
            "gflops": None,
            "params": total_params,
            "params_m": total_params / 1e6,
        }


def profile_dit_model(dit_config: dict, batch_size: int = 1, device: str = "cuda") -> None:
    """
    Profile a DiT model configuration.

    Parameters
    ----------
    dit_config : dict
        Configuration dictionary for DiT model
    batch_size : int
        Batch size for profiling (default: 1)
    device : str
        Device to run on (default: 'cuda')
    """
    from tqdne.dit import DiT

    print(f"\nProfiling DiT model with config:")
    for key, val in dit_config.items():
        print(f"  {key}: {val}")

    # Create model
    model = DiT(**dit_config).to(device).eval()

    # Profile static FLOPs
    input_size = dit_config.get("input_size", 32)
    in_channels = dit_config.get("in_channels", 8)
    cond_features = dit_config.get("cond_features", 5)

    input_shape = (batch_size, in_channels, input_size, input_size)

    print(f"\nStatic profiling with input shape: {input_shape}")
    results = profile_model_static(model, input_shape, device)

    print(f"\nModel Statistics:")
    print(f"  Parameters: {results['params_m']:.2f}M")
    if results['gflops'] is not None:
        print(f"  FLOPs per forward pass: {results['gflops']:.2f} GFLOPs")
        print(f"  FLOPs per sample (batch={batch_size}): {results['gflops']/batch_size:.2f} GFLOPs")

    # Profile dynamic execution
    print(f"\nDynamic profiling (actual execution)...")
    counter = FLOPsCounter()

    with th.no_grad():
        dummy_x = th.randn(input_shape).to(device)
        dummy_t = th.randn(batch_size).to(device)
        dummy_cond = th.randn(batch_size, cond_features).to(device)

        # Warmup
        for _ in range(3):
            _ = model(dummy_x, dummy_t, dummy_cond)

        # Profile
        for _ in range(10):
            with counter.profile():
                _ = model(dummy_x, dummy_t, dummy_cond)

    counter.print_summary("DiT Forward Pass Profiling")


def profile_generation(
    model,
    shape: tuple,
    cond: th.Tensor | None = None,
    num_samples: int = 1,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Profile waveform generation including sampling loop.

    Parameters
    ----------
    model : LightningModule
        The generative model (LightningDiT, LightningEDM, etc.)
    shape : tuple
        Shape of samples to generate
    cond : torch.Tensor, optional
        Conditional features
    num_samples : int
        Number of samples to generate (default: 1)
    device : str
        Device to run on (default: 'cuda')

    Returns
    -------
    dict
        Dictionary with timing and throughput metrics
    """
    counter = FLOPsCounter()

    print(f"\nProfiling generation of {num_samples} samples...")
    print(f"Sample shape: {shape}")

    start_total = time.perf_counter()

    for i in range(num_samples):
        with counter.profile():
            with th.no_grad():
                _ = model.sample(shape, cond=cond[i:i+1] if cond is not None else None)

    end_total = time.perf_counter()
    total_time = end_total - start_total

    counter.print_summary("Generation Profiling")

    print(f"Total generation time: {total_time:.2f} s")
    print(f"Time per sample: {total_time/num_samples:.2f} s")
    print(f"Samples per second: {num_samples/total_time:.2f}")

    return {
        "total_time": total_time,
        "time_per_sample": total_time / num_samples,
        "samples_per_second": num_samples / total_time,
        "total_gflops": counter.gflops,
        "gflops_per_second": counter.gflops_per_second,
    }
