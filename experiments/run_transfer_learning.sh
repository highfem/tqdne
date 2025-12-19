#!/bin/bash
# Transfer Learning Workflow Script for TQDNE
# This script demonstrates a complete transfer learning workflow

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Source domain paths (pre-trained models)
SOURCE_WORKDIR="/path/to/source_domain_workdir"
SOURCE_AUTOENCODER="$SOURCE_WORKDIR/outputs/Autoencoder-32x32x4-LogSpectrogram/last.ckpt"
SOURCE_DIFFUSION="$SOURCE_WORKDIR/outputs/Latent-DiT-32x32x8-LogSpectrogram/last.ckpt"

# Target domain paths (new location)
TARGET_WORKDIR="/path/to/target_domain_workdir"

# Transfer settings
STRATEGY="conservative"  # Options: "conservative" or "aggressive"
FREEZE_AUTOENCODER_PHASE2=true  # Freeze autoencoder in Phase 2

# Training hyperparameters
NUM_DEVICES=4  # Number of GPUs
NUM_WORKERS=32  # Number of data loader workers
AUTOENCODER_BATCH_SIZE=128
DIFFUSION_BATCH_SIZE=256

# Optional: Override default training steps and learning rates
# AUTOENCODER_LR="1e-5"
# AUTOENCODER_STEPS="30000"
# DIFFUSION_LR="1e-5"
# DIFFUSION_STEPS="100000"

# ============================================================================
# VALIDATION
# ============================================================================

echo "========================================="
echo "TQDNE Transfer Learning Workflow"
echo "========================================="
echo ""

# Check if source checkpoints exist
if [ ! -f "$SOURCE_AUTOENCODER" ]; then
    echo "ERROR: Source autoencoder checkpoint not found: $SOURCE_AUTOENCODER"
    exit 1
fi

if [ ! -f "$SOURCE_DIFFUSION" ]; then
    echo "ERROR: Source diffusion checkpoint not found: $SOURCE_DIFFUSION"
    exit 1
fi

# Check if target data exists
if [ ! -f "$TARGET_WORKDIR/data/preprocessed_waveforms.h5" ]; then
    echo "ERROR: Target domain data not found: $TARGET_WORKDIR/data/preprocessed_waveforms.h5"
    echo "Please prepare your target domain data first."
    exit 1
fi

echo "Configuration:"
echo "  Source workdir: $SOURCE_WORKDIR"
echo "  Target workdir: $TARGET_WORKDIR"
echo "  Strategy: $STRATEGY"
echo "  GPUs: $NUM_DEVICES"
echo ""

# ============================================================================
# PHASE 1: TRANSFER AUTOENCODER
# ============================================================================

echo "========================================="
echo "PHASE 1: Transferring Autoencoder"
echo "========================================="
echo ""

AUTOENCODER_ARGS=(
    --workdir "$SOURCE_WORKDIR"
    --target-workdir "$TARGET_WORKDIR"
    --source-checkpoint "$SOURCE_AUTOENCODER"
    --strategy "$STRATEGY"
    --batchsize "$AUTOENCODER_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --num-devices "$NUM_DEVICES"
)

# Add optional learning rate override
if [ ! -z "$AUTOENCODER_LR" ]; then
    AUTOENCODER_ARGS+=(--learning-rate "$AUTOENCODER_LR")
fi

# Add optional steps override
if [ ! -z "$AUTOENCODER_STEPS" ]; then
    AUTOENCODER_ARGS+=(--max-steps "$AUTOENCODER_STEPS")
fi

echo "Running: python experiments/transfer_autoencoder.py ${AUTOENCODER_ARGS[@]}"
echo ""

python experiments/transfer_autoencoder.py "${AUTOENCODER_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 1 (Autoencoder transfer) failed!"
    exit 1
fi

echo ""
echo "Phase 1 completed successfully!"
echo ""

# ============================================================================
# PHASE 2: TRANSFER DIFFUSION MODEL
# ============================================================================

echo "========================================="
echo "PHASE 2: Transferring Diffusion Model"
echo "========================================="
echo ""

DIFFUSION_ARGS=(
    --workdir "$SOURCE_WORKDIR"
    --target-workdir "$TARGET_WORKDIR"
    --source-autoencoder-checkpoint "$SOURCE_AUTOENCODER"
    --source-diffusion-checkpoint "$SOURCE_DIFFUSION"
    --strategy "$STRATEGY"
    --batchsize "$DIFFUSION_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --num-devices "$NUM_DEVICES"
)

# Add freeze-autoencoder flag if enabled
if [ "$FREEZE_AUTOENCODER_PHASE2" = true ]; then
    DIFFUSION_ARGS+=(--freeze-autoencoder)
fi

# Add optional learning rate override
if [ ! -z "$DIFFUSION_LR" ]; then
    DIFFUSION_ARGS+=(--learning-rate "$DIFFUSION_LR")
fi

# Add optional steps override
if [ ! -z "$DIFFUSION_STEPS" ]; then
    DIFFUSION_ARGS+=(--max-steps "$DIFFUSION_STEPS")
fi

echo "Running: python experiments/transfer_diffusion.py ${DIFFUSION_ARGS[@]}"
echo ""

python experiments/transfer_diffusion.py "${DIFFUSION_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 2 (Diffusion transfer) failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "TRANSFER LEARNING COMPLETE!"
echo "========================================="
echo ""
echo "Output checkpoints:"
echo "  Autoencoder: $TARGET_WORKDIR/outputs/Transfer-Autoencoder-32x32x4-LogSpectrogram/"
echo "  Diffusion:   $TARGET_WORKDIR/outputs/Transfer-DiT-32x32x8-LogSpectrogram/"
echo ""
echo "Next steps:"
echo "  1. Evaluate the transferred models on target domain test data"
echo "  2. Generate synthetic waveforms using the transferred diffusion model"
echo "  3. Compare generated samples with real target domain data"
echo ""
