#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=0-11:58:00
#SBATCH --job-name=openr1-sft-qwen3-8b
#SBATCH --output=/scratch/zainab14/open-r1/logs/%x-%j.out
#SBATCH --error=/scratch/zainab14/open-r1/logs/%x-%j.err

set -euo pipefail

echo "Job started on $(hostname) at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

SCRATCH_DIR=/scratch/zainab14
PROJECT_DIR=$SCRATCH_DIR/open-r1

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

echo "--- Loading modules ---"
module purge
module load python/3.11

echo "--- GPU info ---"
nvidia-smi || echo "nvidia-smi failed"

echo "--- Activating uv-created virtual environment ---"
source openr1/bin/activate
which python
python --version

# export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=$SCRATCH_DIR/hf_cache
export XDG_CACHE_HOME=$SCRATCH_DIR/.cache
export TRITON_CACHE_DIR=$SCRATCH_DIR/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH_DIR/.inductor_cache
export VLLM_CACHE_ROOT=$SCRATCH_DIR/.vllm_cache
export XDG_CONFIG_HOME=$SCRATCH_DIR/.config
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CONFIG_HOME"

export MPLCONFIGDIR=$SCRATCH_DIR/.mplconfig
export FLASHINFER_WORKSPACE_BASE=$SCRATCH_DIR
export VLLM_CONFIG_ROOT=$SCRATCH_DIR/.vllm_config
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
mkdir -p "$MPLCONFIGDIR" "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CONFIG_ROOT"
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH_DIR/wandb
export WANDB_CACHE_DIR=$SCRATCH_DIR/.wandb_cache
export WANDB_CONFIG_DIR=$SCRATCH_DIR/.wandb_config
export WANDB_DATA_DIR=$SCRATCH_DIR/.wandb_data
export WANDB_ARTIFACT_DIR=$SCRATCH_DIR/.wandb_artifacts
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
export PYTHONPATH=$PROJECT_DIR/src:${PYTHONPATH:-}

CONFIG=recipes/Qwen3-8B/sft/config.yaml
echo "--- Running SFT with $CONFIG ---"
python src/open_r1/sft.py --config "$CONFIG"

echo "Job finished at: $(date)"