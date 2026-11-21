#!/bin/bash
#SBATCH --job-name=mesa-bcg-models
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/rdb200/nsrr-setup/logs/mesa-bcg-models-%j.out
#SBATCH --error=/home/rdb200/nsrr-setup/logs/mesa-bcg-models-%j.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="/home/rdb200/nsrr-setup/.deps-bcg${PYTHONPATH:+:$PYTHONPATH}"
exec .venv-transformer/bin/python -u bcg_jitter_staging/benchmark.py "$@"
