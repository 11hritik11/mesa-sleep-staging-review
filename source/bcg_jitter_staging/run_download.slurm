#!/bin/bash
#SBATCH --job-name=mesa-rpoints
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rdb200/nsrr-setup/logs/mesa-rpoints-%j.out
#SBATCH --error=/home/rdb200/nsrr-setup/logs/mesa-rpoints-%j.err
set -euo pipefail
cd /home/rdb200/nsrr-setup
export REQUESTS_CA_BUNDLE=/home/rdb200/nsrr-setup/.cache/nsrr-ca-chain.pem
exec .venv-transformer/bin/python -u bcg_jitter_staging/download_annotations.py "$@"
