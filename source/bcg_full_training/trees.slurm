#!/bin/bash
#SBATCH --job-name=mesa-full-trees
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/mesa-full-trees-%j.out
#SBATCH --error=logs/mesa-full-trees-%j.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export PYTHONPATH=/home/rdb200/nsrr-setup/.deps-bcg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
.venv-transformer/bin/python bcg_full_training/verify_source.py "$1/source_checksums.json"
.venv-transformer/bin/python -u bcg_full_training/trees.py --prepared "$1/prepared" --output "$1/trees" --threads 8
