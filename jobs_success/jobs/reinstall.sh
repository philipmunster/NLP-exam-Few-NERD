#!/bin/bash
#SBATCH --job-name=reinstall_packages
#SBATCH --output=/home/pmyn/nlp-exam/jobs/reinstall.out
#SBATCH --error=/home/pmyn/nlp-exam/jobs/reinstall.err
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr

set -e
set -x

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
module load CUDA/12.1.1

export TMPDIR=$HOME/tmp
export CONDA_PKGS_DIRS=$HOME/tmp/conda_cache

# Install pip into the env using conda directly
/opt/itu/easybuild/software/Miniconda3/25.5.1-1/bin/conda install -y \
    -p $HOME/nlp-exam/envs/nlp_env pip

# Now use pip directly from the env
$HOME/nlp-exam/envs/nlp_env/bin/pip install --no-user torch --index-url https://download.pytorch.org/whl/cu121
$HOME/nlp-exam/envs/nlp_env/bin/pip install --no-user -r $HOME/nlp-exam/NLP-exam-Few-NERD/requirements.txt
$HOME/nlp-exam/envs/nlp_env/bin/pip install --no-user bitsandbytes