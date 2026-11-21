#!/bin/bash
#SBATCH --job-name=mesa-bcg-jitter
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/rdb200/nsrr-setup/logs/mesa-bcg-jitter-%j.out
#SBATCH --error=/home/rdb200/nsrr-setup/logs/mesa-bcg-jitter-%j.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv-transformer/bin/python -u bcg_jitter_staging/experiment.py "$@"
