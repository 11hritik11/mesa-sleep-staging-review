#!/bin/bash
#SBATCH --job-name=mesa-balanced-deep
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%3
#SBATCH --output=logs/mesa-balanced-deep-%A_%a.out
#SBATCH --error=logs/mesa-balanced-deep-%A_%a.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export PYTHONPATH=/home/rdb200/nsrr-setup/.deps-bcg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
.venv-transformer/bin/python bcg_full_training/verify_source.py "$1/source_checksums.json"
.venv-transformer/bin/python -u bcg_balanced_staging/train_balanced_deep.py --prepared "$1/prepared" --output "$1/deep" --index "$SLURM_ARRAY_TASK_ID" --threads 8
