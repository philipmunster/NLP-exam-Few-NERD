#!/bin/bash
#SBATCH --job-name=verify_env
#SBATCH --output=/home/pmyn/nlp-exam/jobs/verify.out
#SBATCH --error=/home/pmyn/nlp-exam/jobs/verify.err
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr

set -e
set -x

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
module load CUDA/12.1.1

export TMPDIR=$HOME/tmp

$HOME/nlp-exam/envs/nlp_env/bin/python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

import transformers
print("Transformers version:", transformers.__version__)

import peft
print("PEFT version:", peft.__version__)

import bitsandbytes
print("Bitsandbytes version:", bitsandbytes.__version__)

print("ALL OK")
EOF