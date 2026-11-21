#!/bin/bash
#SBATCH --job-name=mesa-balanced-report
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mesa-balanced-report-%j.out
#SBATCH --error=logs/mesa-balanced-report-%j.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export PYTHONPATH=/home/rdb200/nsrr-setup/.deps-bcg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
.venv-transformer/bin/python bcg_full_training/verify_source.py "$1/source_checksums.json"
.venv-transformer/bin/python -u bcg_balanced_staging/finalize_balanced.py --prepared "$1/prepared" --classic "$1/classic" --deep "$1/deep" --controls "$1/controls" --output "$1/results" --threads 8
