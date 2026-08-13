#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=0-5:58:00
#SBATCH --output=/scratch/zainab14/open-r1/logs/%x-%j.out
#SBATCH --error=/scratch/zainab14/open-r1/logs/%x-%j.err

set -euo pipefail

MODEL=${1:?usage: sbatch --job-name=<name> slurm/script_runner_cc_trillium_run_eval.sh <model> <task> <tag>}
TASK=${2:?}
TAG=${3:?}

echo "Job started on $(hostname) at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Task: $TASK"

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

export PYTHONPATH=$PROJECT_DIR/src:${PYTHONPATH:-}

OUTPUT_DIR=$PROJECT_DIR/outputs/eval/$TAG/$TASK
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:16384,temperature:0.6,top_p:0.95,top_k:20}"

echo "--- Running lighteval: $TAG | $TASK ---"
lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    --save-details

echo "Job finished at: $(date)"