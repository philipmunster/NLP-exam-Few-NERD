#!/bin/bash
#SBATCH --job-name=fix_torch
#SBATCH --output=/home/pmyn/nlp-exam/jobs/fix_torch.out
#SBATCH --error=/home/pmyn/nlp-exam/jobs/fix_torch.err
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr

set -e
set -x

module load CUDA/12.1.1
module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0

export TMPDIR=$HOME/tmp

$HOME/nlp-exam/envs/nlp_env/bin/pip install --no-user \
    torch==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121