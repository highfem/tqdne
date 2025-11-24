#!/usr/bin/env python
"""Profile GFLOPs usage during waveform generation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdne.autoencoder import LightningAutoencoder
from tqdne.dit import LightningDiT
from tqdne.edm import LightningEDM
from tqdne.profiling import FLOPsCounter, profile_generation, profile_model_static
from tqdne.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Profile FLOPs during waveform generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (EDM or DiT)",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        default=None,
        help="Path to autoencoder checkpoint (for latent models)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate for profiling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--static_only",
        action="store_true",
        help="Only do static profiling (no generation)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load autoencoder if provided
    autoencoder = None
    if args.autoencoder_checkpoint:
        print(f"Loading autoencoder from {args.autoencoder_checkpoint}")
        autoencoder = (
            LightningAutoencoder.load_from_checkpoint(args.autoencoder_checkpoint)
            .to(device)
            .eval()
        )

        # Profile autoencoder
        print("\n" + "="*60)
        print("AUTOENCODER PROFILING")
        print("="*60)
        enc_results = profile_model_static(
            autoencoder.encoder, (args.batch_size, 3, 128, 128), device
        )
        dec_results = profile_model_static(
            autoencoder.decoder, (args.batch_size, 8, 32, 32), device
        )

        print(f"\nEncoder:")
        print(f"  Parameters: {enc_results['params_m']:.2f}M")
        if enc_results['gflops']:
            print(f"  GFLOPs per forward: {enc_results['gflops']:.2f}")

        print(f"\nDecoder:")
        print(f"  Parameters: {dec_results['params_m']:.2f}M")
        if dec_results['gflops']:
            print(f"  GFLOPs per forward: {dec_results['gflops']:.2f}")

    # Load main model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)

    # Try loading as DiT first, then EDM
    try:
        model = (
            LightningDiT.load_from_checkpoint(checkpoint_path, autoencoder=autoencoder)
            .to(device)
            .eval()
        )
        model_type = "DiT"
    except Exception:
        model = (
            LightningEDM.load_from_checkpoint(checkpoint_path, autoencoder=autoencoder)
            .to(device)
            .eval()
        )
        model_type = "EDM"

    print(f"Loaded {model_type} model")

    # Profile model architecture
    print("\n" + "="*60)
    print(f"{model_type} MODEL PROFILING")
    print("="*60)

    if model_type == "DiT":
        input_shape = (args.batch_size, model.dit.in_channels, 32, 32)
        results = profile_model_static(model.dit, input_shape, device)
    else:
        input_shape = (args.batch_size, 8, 32, 32)  # Assuming latent space
        results = profile_model_static(model.model, input_shape, device)

    print(f"\n{model_type} Network:")
    print(f"  Parameters: {results['params_m']:.2f}M")
    if results['gflops']:
        print(f"  GFLOPs per forward: {results['gflops']:.2f}")
        print(f"  GFLOPs per sample: {results['gflops']/args.batch_size:.2f}")

    if args.static_only:
        return

    # Profile generation (dynamic)
    print("\n" + "="*60)
    print("GENERATION PROFILING (DYNAMIC)")
    print("="*60)

    # Create dummy conditional features
    cond = th.randn(args.num_samples, 5, device=device)

    # Determine shape
    if autoencoder:
        shape = (args.batch_size, 3, 128, 128)
    else:
        shape = (args.batch_size, 8, 32, 32)

    # Profile full generation
    gen_results = profile_generation(
        model,
        shape,
        cond=cond,
        num_samples=args.num_samples,
        device=device,
    )

    # Estimate per-component breakdown
    print("\n" + "="*60)
    print("ESTIMATED BREAKDOWN")
    print("="*60)

    num_steps = model.num_sampling_steps
    print(f"Sampling steps: {num_steps}")

    if results['gflops'] and gen_results['total_gflops']:
        # Rough estimate: 2 model evals per step (Heun's method)
        model_evals_per_sample = num_steps * 2 - 1
        est_model_gflops = results['gflops'] * model_evals_per_sample * args.num_samples

        print(f"\nEstimated GFLOPs per sample:")
        print(f"  Diffusion model: ~{est_model_gflops/args.num_samples:.2f} GFLOPs")

        if autoencoder and enc_results['gflops'] and dec_results['gflops']:
            print(f"  Encoder: ~{enc_results['gflops']:.2f} GFLOPs")
            print(f"  Decoder: ~{dec_results['gflops']:.2f} GFLOPs")
            total_est = est_model_gflops/args.num_samples + enc_results['gflops'] + dec_results['gflops']
            print(f"  Total per sample: ~{total_est:.2f} GFLOPs")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Time per sample: {gen_results['time_per_sample']:.3f} s")
    print(f"Throughput: {gen_results['samples_per_second']:.2f} samples/s")
    if gen_results['total_gflops']:
        print(f"Total GFLOPs: {gen_results['total_gflops']:.2f}")
        print(f"GFLOPS/s: {gen_results['gflops_per_second']:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
