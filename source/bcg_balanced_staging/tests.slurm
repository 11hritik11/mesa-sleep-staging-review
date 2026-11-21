#!/bin/bash
#SBATCH --job-name=mesa-balanced-tests
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mesa-balanced-tests-%j.out
#SBATCH --error=logs/mesa-balanced-tests-%j.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export PYTHONPATH=/home/rdb200/nsrr-setup/.deps-bcg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MPLBACKEND=Agg
.venv-transformer/bin/python -u bcg_balanced_staging/tests_balanced.py --real
