# Transfer Learning Guide for TQDNE

This guide explains how to use transfer learning to adapt pre-trained seismic waveform generation models to new geographic locations with different data.

## Overview

Transfer learning allows you to leverage existing trained models from one seismic region (source domain) and adapt them to generate realistic waveforms for a different region (target domain). This is especially valuable when you have limited data in the new region.

## Two-Phase Transfer Learning Approach

TQDNE uses a two-phase approach:

1. **Phase 1: Autoencoder Transfer** - Adapt the VAE autoencoder to the target domain's signal characteristics
2. **Phase 2: Diffusion Model Transfer** - Fine-tune the generative diffusion model (DiT or EDM) using the adapted autoencoder

## Prerequisites

### Source Domain Requirements

You need access to:
- Pre-trained autoencoder checkpoint (e.g., `Autoencoder-32x32x4-LogSpectrogram/last.ckpt`)
- Pre-trained diffusion model checkpoint (e.g., `Latent-DiT-32x32x8-LogSpectrogram/last.ckpt`)

### Target Domain Requirements

Your target domain data must match the source domain format:
- HDF5 file: `preprocessed_waveforms.h5`
- Same structure as source data:
  - `waveforms`: (N, 3, T) - 3-channel seismograms
  - `normalized_features`: (N, 5) - Conditional features
  - Same feature keys: `magnitude`, `hypocentral_distance`, `vs30`, `hypocentre_depth`, `azimuthal_gap`
- Same sampling rate: 100 Hz
- Same waveform length: 4064 samples (40.64 seconds)

## Transfer Strategies

### Conservative Strategy (Recommended)

**Best for:** Target domain similar to source domain (same instruments, similar geology)

**Settings:**
- Freeze encoder, fine-tune decoder
- Learning rate: 1e-5 (10x lower than original)
- Training steps: 30k (autoencoder), 100k (diffusion)

**Use when:**
- You have limited target domain data (< 5000 samples)
- Source and target domains have similar seismic characteristics
- You want faster convergence with less risk of overfitting

### Aggressive Strategy

**Best for:** Target domain significantly different from source domain

**Settings:**
- Fine-tune all parameters
- Learning rate: 1e-6 (100x lower than original)
- Training steps: Same as conservative but may need more

**Use when:**
- Target domain has very different characteristics (different rock types, depths, magnitudes)
- You have substantial target domain data (> 10,000 samples)
- Conservative approach shows poor reconstruction quality

## Usage

### Phase 1: Transfer Autoencoder

```bash
python experiments/transfer_autoencoder.py \
  --workdir /path/to/source_domain_workdir \
  --target-workdir /path/to/target_domain_workdir \
  --source-checkpoint /path/to/source_domain_workdir/outputs/Autoencoder-32x32x4-LogSpectrogram/last.ckpt \
  --strategy conservative \
  --batchsize 128 \
  --num-workers 32 \
  --num-devices 4
```

**Key Arguments:**
- `--workdir`: Source domain working directory (reference only, not modified)
- `--target-workdir`: Target domain working directory (where new data and outputs are saved)
- `--source-checkpoint`: Path to source autoencoder checkpoint
- `--strategy`: Transfer strategy (`conservative` or `aggressive`)
- `--num-devices`: Number of GPUs to use (4 recommended for multi-GPU training)

**Optional Overrides:**
- `--freeze-encoder`: Explicitly freeze/unfreeze encoder (overrides strategy default)
- `--freeze-decoder`: Explicitly freeze/unfreeze decoder
- `--learning-rate`: Custom learning rate
- `--max-steps`: Custom maximum training steps
- `--name`: Custom experiment name
- `--resume`: Resume from last checkpoint if training was interrupted

**Output:**
Saved to `target_workdir/outputs/Transfer-Autoencoder-32x32x4-LogSpectrogram/`

**Expected Duration:**
- Conservative: ~2-4 hours on 4x A100 GPUs
- Aggressive: ~3-6 hours on 4x A100 GPUs

### Phase 2: Transfer Diffusion Model

After Phase 1 completes successfully:

```bash
python experiments/transfer_diffusion.py \
  --workdir /path/to/source_domain_workdir \
  --target-workdir /path/to/target_domain_workdir \
  --source-autoencoder-checkpoint /path/to/source_domain_workdir/outputs/Autoencoder-32x32x4-LogSpectrogram/last.ckpt \
  --source-diffusion-checkpoint /path/to/source_domain_workdir/outputs/Latent-DiT-32x32x8-LogSpectrogram/last.ckpt \
  --strategy conservative \
  --freeze-autoencoder \
  --batchsize 256 \
  --num-workers 32 \
  --num-devices 4
```

**Key Arguments:**
- `--source-autoencoder-checkpoint`: Original source autoencoder (reference)
- `--source-diffusion-checkpoint`: Source diffusion model to transfer
- `--transferred-autoencoder-checkpoint`: (Optional) Path to Phase 1 output. Auto-detected if not specified.
- `--freeze-autoencoder`: Recommended - freezes autoencoder to focus on diffusion training
- `--model-type`: Specify `dit` or `edm` if not auto-detected from checkpoint path

**Output:**
Saved to `target_workdir/outputs/Transfer-DiT-32x32x8-LogSpectrogram/` or `Transfer-EDM-...`

**Expected Duration:**
- Conservative: ~8-12 hours on 4x A100 GPUs
- Aggressive: ~12-20 hours on 4x A100 GPUs

## Example Workflow

### Complete Example: California → New Zealand

Assume you have:
- Source domain trained on California seismic data
- Target domain with New Zealand seismic data (limited samples)

```bash
# Setup
export SOURCE_DIR=/data/california_seismic
export TARGET_DIR=/data/newzealand_seismic

# Step 1: Prepare target domain data
# Ensure your target HDF5 file is at: $TARGET_DIR/data/preprocessed_waveforms.h5

# Step 2: Transfer autoencoder
python experiments/transfer_autoencoder.py \
  --workdir $SOURCE_DIR \
  --target-workdir $TARGET_DIR \
  --source-checkpoint $SOURCE_DIR/outputs/Autoencoder-32x32x4-LogSpectrogram/last.ckpt \
  --strategy conservative \
  -b 128 -w 32 -d 4

# Wait for Phase 1 to complete (~3 hours)

# Step 3: Transfer diffusion model
python experiments/transfer_diffusion.py \
  --workdir $SOURCE_DIR \
  --target-workdir $TARGET_DIR \
  --source-autoencoder-checkpoint $SOURCE_DIR/outputs/Autoencoder-32x32x4-LogSpectrogram/last.ckpt \
  --source-diffusion-checkpoint $SOURCE_DIR/outputs/Latent-DiT-32x32x8-LogSpectrogram/last.ckpt \
  --strategy conservative \
  --freeze-autoencoder \
  -b 256 -w 32 -d 4

# Wait for Phase 2 to complete (~10 hours)

# Step 4: Generate samples with transferred model
# Use the same inference scripts with the new checkpoint path
```

## Monitoring Training

Both scripts log to Weights & Biases (wandb) by default.

**Key Metrics to Monitor:**

### Phase 1 (Autoencoder):
- `validation/loss`: Should decrease steadily
- `validation/mse_channel_*`: Reconstruction quality per channel
- `validation/asd_channel_*`: Spectral density matching

**Good signs:**
- Validation loss plateaus after 10k-20k steps
- Reconstructed waveforms look visually similar to targets
- ASD curves match between real and reconstructed

**Warning signs:**
- Validation loss increasing (overfitting - reduce learning rate)
- Blurry reconstructions (need more steps or unfreeze encoder)
- Very different ASD curves (may need aggressive strategy)

### Phase 2 (Diffusion):
- `validation/loss`: Denoising loss should decrease
- `validation/asd_channel_*`: Generated samples should match target distribution
- Sample plots: Visual quality of generated waveforms

**Good signs:**
- Loss decreases in first 20k-30k steps, then plateaus
- Generated waveforms have realistic characteristics
- ASD curves match target domain statistics

**Warning signs:**
- Loss not decreasing (learning rate too high/low)
- Generated samples look like source domain, not target (need more steps)
- Mode collapse (samples lack diversity - may need to adjust hyperparameters)

## Advanced Configuration

### Custom Transfer Settings

You can create custom transfer configurations by modifying `experiments/config.py`:

```python
from config import TransferConfig

# Create custom config
my_config = TransferConfig(
    workdir="source_workdir",
    target_workdir="target_workdir",
    transfer_strategy="conservative",
    autoencoder_learning_rate=5e-6,  # Even lower LR
    autoencoder_max_steps=50_000,     # More steps
    diffusion_learning_rate=5e-6,
    diffusion_max_steps=150_000,
    freeze_encoder=True,
    freeze_decoder=False,
)
```

### Resuming Interrupted Training

If training is interrupted, resume with `--resume`:

```bash
python experiments/transfer_autoencoder.py \
  --workdir $SOURCE_DIR \
  --target-workdir $TARGET_DIR \
  --source-checkpoint $SOURCE_DIR/outputs/Autoencoder-32x32x4-LogSpectrogram/last.ckpt \
  --strategy conservative \
  --resume  # <-- Loads last checkpoint
```

### Multiple GPU Configuration

Adjust `--num-devices` based on available GPUs:

```bash
# Single GPU
--num-devices 1

# 4 GPUs (recommended)
--num-devices 4

# 8 GPUs (for very large datasets)
--num-devices 8
```

PyTorch Lightning automatically uses Distributed Data Parallel (DDP) for multi-GPU training.

## Troubleshooting

### Issue: "Checkpoint not found"
**Solution:** Verify checkpoint paths are correct and files exist. Use absolute paths.

### Issue: "Target data format mismatch"
**Solution:** Ensure target HDF5 has same structure as source. Check:
```python
import h5py
with h5py.File("target_workdir/data/preprocessed_waveforms.h5", "r") as f:
    print(f.keys())
    print(f["waveforms"].shape)  # Should be (N, 3, 4064)
    print(f["normalized_features"].shape)  # Should be (N, 5)
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
- Phase 1: Try `--batchsize 64` or `--batchsize 32`
- Phase 2: Try `--batchsize 128` or `--batchsize 64`

### Issue: "Validation loss not decreasing"
**Solutions:**
1. Try aggressive strategy: `--strategy aggressive`
2. Increase training steps: `--max-steps 50000` (Phase 1) or `--max-steps 150000` (Phase 2)
3. Adjust learning rate: `--learning-rate 1e-6` (lower) or `--learning-rate 5e-5` (higher)

### Issue: "Generated samples don't look realistic"
**Solutions:**
1. Phase 1: Check autoencoder reconstruction quality first
2. Phase 2: Train longer (increase `--max-steps`)
3. Try unfreezing autoencoder (remove `--freeze-autoencoder` flag)
4. Check if target data is properly normalized

### Issue: "Training is too slow"
**Solutions:**
1. Increase number of GPUs: `--num-devices 8`
2. Increase batch size (if memory allows): `--batchsize 256` or `--batchsize 512`
3. Reduce number of workers if I/O bottleneck: `--num-workers 16`

## Best Practices

1. **Always run Phase 1 before Phase 2** - The diffusion model depends on a well-adapted autoencoder

2. **Validate autoencoder quality** - Visually inspect reconstructions before proceeding to Phase 2

3. **Start conservative** - Use conservative strategy first, only switch to aggressive if results are poor

4. **Monitor validation metrics** - Don't just watch training loss, check validation and sample quality

5. **Save checkpoints frequently** - Default is every 5k/10k steps, which is good for experimentation

6. **Use frozen autoencoder in Phase 2** - Recommended unless autoencoder quality is poor

7. **Match source normalization** - Target data must use same preprocessing as source (STFT params, normalization)

8. **Sufficient target data** - Aim for at least 1000-2000 target domain samples for good results

## Flow Matching Transfer Learning

### Overview

In addition to transferring DiT and EDM diffusion models, TQDNE supports transfer learning for Flow Matching models. This includes both standard transfer (FlowMatching → FlowMatching) and cross-transfer (DiT diffusion → FlowMatching).

### Usage

#### Standard Transfer (FlowMatching → FlowMatching)

```bash
python experiments/transfer_flow_matching.py \
  --workdir /path/to/source_domain_workdir \
  --target-workdir /path/to/target_domain_workdir \
  --source-autoencoder-checkpoint /path/to/source/outputs/Autoencoder-.../last.ckpt \
  --source-flow-matching-checkpoint /path/to/source/outputs/Latent-FlowMatching-.../last.ckpt \
  --strategy conservative \
  --freeze-autoencoder \
  -b 256 -w 32 -d 4
```

#### Cross-Transfer (DiT Diffusion → FlowMatching)

```bash
python experiments/transfer_flow_matching.py \
  --workdir /path/to/source_domain_workdir \
  --target-workdir /path/to/target_domain_workdir \
  --source-autoencoder-checkpoint /path/to/source/outputs/Autoencoder-.../last.ckpt \
  --source-diffusion-checkpoint /path/to/source/outputs/Latent-DiT-.../last.ckpt \
  --strategy conservative \
  --freeze-autoencoder \
  --model-channels 768 \
  -b 256 -w 32 -d 4
```

### Why Cross-Transfer from DiT to FlowMatching Works

#### Mathematical Formulations

**DiT Diffusion Model:**
- Forward process: x_noisy = x_clean + σ·ε, where σ ~ lognormal, ε ~ N(0,I)
- Conditioning: 0.25 * log(σ)
- Prediction: Denoised output with EDM skip connections
- Loss: Sigma-weighted MSE

**FlowMatching Model:**
- Forward process: x_t = t·x_1 + (1-t)·x_0, where t ~ U[0,1], x_0 ~ N(0,I)
- Conditioning: t * 999.0
- Prediction: Velocity field v = x_1 - x_0
- Loss: Unweighted MSE

#### Theoretical Connection

Both diffusion and flow matching are **continuous-time generative models** based on ODEs:

1. **Diffusion models** have a deterministic formulation called the "probability flow ODE"
2. **Flow matching** uses continuous normalizing flows with learned velocity fields
3. Both frameworks generate samples by integrating an ODE from noise to data

While the objectives differ, they solve fundamentally related problems: learning to transform noise into data through continuous-time dynamics.

#### Why Transfer Learning Works

**Shared Knowledge:**
- **Latent space structure:** DiT learned to understand seismic waveform spectrogram features
- **Conditional dependencies:** Both models condition on the same earthquake parameters (magnitude, distance, depth, etc.)
- **Semantic representations:** Lower DiT transformer layers learn general features that are transferable across related tasks

**Architecture Flexibility:**
- DiT uses sinusoidal timestep embeddings that can handle any scalar conditioning input
- The transformer architecture is task-agnostic—it processes embeddings generically
- Fine-tuning adapts the model from denoising (diffusion) to velocity prediction (flow matching)

**Analogy to Computer Vision:**
Similar to how ImageNet-pretrained vision transformers transfer to segmentation tasks:
- Different objectives (classification vs. segmentation)
- But pretrained visual features provide strong initialization
- Fine-tuning adapts features to the new task

#### Important Clarifications

**This is NOT mathematical equivalence** — We are not performing an exact transformation of diffusion weights into flow matching weights. The two models have different training objectives and prediction targets.

**This IS principled transfer learning** — We leverage learned latent space representations as initialization for a related but different generative modeling task.

**This IS theoretically justified** — Both frameworks solve continuous-time generative modeling via ordinary differential equations. Recent research (e.g., "Flow Matching for Generative Modeling", Lipman et al. 2023) shows that diffusion and flow matching are related approaches to the same fundamental problem.

#### What Gets Adapted During Fine-Tuning

When cross-transferring from DiT to FlowMatching, the model adapts:

1. **Conditioning distribution:** Learns to use t ~ Uniform[0,1] instead of σ ~ lognormal
2. **Prediction target:** Adapts from predicting denoised samples to predicting velocity fields
3. **Loss weighting:** Learns with unweighted MSE instead of sigma-weighted loss
4. **Output processing:** Removes dependency on EDM skip connections

The DiT transformer backbone provides a strong initialization with learned features, which fine-tuning then adapts to the flow matching objective.

#### Expected Performance

Cross-transfer from DiT to FlowMatching typically provides:

- **Faster convergence** than training flow matching from scratch (pretrained features provide good initialization)
- **Better sample quality** with limited target data (transfer learning benefit)
- **Requires adequate fine-tuning** (recommended: longer warmup and more steps than standard transfer)

#### Recommended Hyperparameters

**For Cross-Transfer (DiT → FlowMatching):**

Conservative strategy:
- Learning rate: 1e-5
- Warmup steps: 1000 (longer than standard 500)
- Max steps: 150,000 (more than standard 100,000)
- Freeze autoencoder: Recommended

Aggressive strategy:
- Learning rate: 1e-6
- Warmup steps: 1000
- Max steps: 150,000

**Rationale:** Cross-transfer requires longer warmup to stabilize the new conditioning scheme and more training steps for full adaptation to the velocity prediction objective.

## Citation

If you use this transfer learning implementation, please cite the original TQDNE paper:

```
Palgunadi, K. H., Bergmeister, A., Bosisio, A., Ermert, L. A., Koroni, M., Perraudin, N., et al. (2025).
High resolution seismic waveform generation using denoising diffusion.
Journal of Geophysical Research: Machine Learning and Computation, 2, e2025JH000862.
https://doi.org/10.1029/2025JH000862
```

## Support

For issues or questions:
- Check the main TQDNE repository documentation
- Review wandb logs for training diagnostics
- Verify data format matches source domain exactly
