#!/bin/bash
#SBATCH --job-name=mesa-full-tcn
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/mesa-full-tcn-%A_%a.out
#SBATCH --error=logs/mesa-full-tcn-%A_%a.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export PYTHONPATH=/home/rdb200/nsrr-setup/.deps-bcg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
variants=(tcn_hrv_common tcn_dual_common tcn_dual_expanded)
.venv-transformer/bin/python bcg_full_training/verify_source.py "$1/source_checksums.json"
.venv-transformer/bin/python -u bcg_full_training/temporal.py --prepared "$1/prepared" --output "$1/temporal/${variants[$SLURM_ARRAY_TASK_ID]}" --variant "${variants[$SLURM_ARRAY_TASK_ID]}" --threads 8
