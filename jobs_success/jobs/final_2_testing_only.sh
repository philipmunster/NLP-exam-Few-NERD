#!/bin/bash
#SBATCH --job-name=final_2_testing_only
#SBATCH --output=/home/pmyn/nlp-exam/jobs/final_2_testing_only_%A_%a.out
#SBATCH --error=/home/pmyn/nlp-exam/jobs/final_2_testing_only_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr
#SBATCH --array=0-1
#SBATCH --mem=40G

set -e
set -x

module load CUDA/12.1.1
module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0

echo "Running on $(hostname)"

JOB_SCRATCH_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}-${SLURM_ARRAY_TASK_ID:-0}"
SCRATCH_DIR=""
for candidate in "/scratch/$JOB_SCRATCH_ID" "/tmp/$USER/$JOB_SCRATCH_ID"; do
  if mkdir -p "$candidate/tmp" 2>/dev/null; then
    SCRATCH_DIR="$candidate"
    break
  fi
done
if [[ -z "$SCRATCH_DIR" ]]; then
  echo "Could not create scratch directory under /scratch or /tmp" >&2
  exit 1
fi
export TMPDIR="$SCRATCH_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME=$HOME/nlp-exam/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
set +x
export HF_TOKEN="$(cat $HOME/.hf_token)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
set -x
mkdir -p "$HF_HOME" "$HOME/nlp-exam/jobs"

cleanup() {
  if [[ -n "${SCRATCH_DIR:-}" && ( "$SCRATCH_DIR" == /scratch/* || "$SCRATCH_DIR" == /tmp/$USER/* ) ]]; then
    echo "Cleaning up scratch directory: $SCRATCH_DIR"
    rm -rf "$SCRATCH_DIR"
  fi
}
trap cleanup EXIT INT TERM

cd $HOME/nlp-exam/NLP-exam-Few-NERD

if [ ! -f data/inter/train.txt ] || [ ! -f data/inter/dev.txt ] || [ ! -f data/inter/test.txt ]; then
  bash data/download.sh inter
fi

# Configurations where training finished but final test checkpoint reload failed.
NS=(5  10)
KS=(5  1)
CKPTS=(checkpoint/proto-inter-5-5-seed0-llama-lora-inter-5way-5shot.pth.tar checkpoint/proto-inter-10-1-seed0-llama-lora-inter-10way-1shot.pth.tar)

N=${NS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}

BATCH_SIZE=1
TEST_ITER=3000
SEED=0

$HOME/nlp-exam/envs/nlp_env/bin/python -u train_demo.py \
  --mode inter \
  --encoder_family llama \
  --encoder_name meta-llama/Llama-3.1-8B \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_bf16 \
  --gradient_checkpointing \
  --model proto \
  --trainN "$N" \
  --N "$N" \
  --K "$K" \
  --Q 1 \
  --batch_size "$BATCH_SIZE" \
  --test_iter "$TEST_ITER" \
  --max_length 64 \
  --seed "$SEED" \
  --load_ckpt "$CKPT" \
  --ckpt_name llama-lora-inter-${N}way-${K}shot \
  --only_test
