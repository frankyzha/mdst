#!/usr/bin/env bash
#SBATCH --job-name=msdt_root_rash_d4
#SBATCH --output=msdt_root_rash_d4_%j.out
#SBATCH --error=msdt_root_rash_d4_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=600G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]

set -euo pipefail

cd /home/users/yz1075/msdt
source .venv/bin/activate
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:${PYTHONPATH:-}

/usr/bin/time -v .venv/bin/python benchmark_root_rashomon_frontier.py \
  --seed 0 \
  --depth 4 \
  --lookahead 3 \
  --max-bins 255 \
  --min-child-size 8 \
  --proposal-atom-cap 32 \
  --max-branching 3 \
  --reg 0.0 \
  --binner-threads 16 \
  --root-time-limit 300 \
  --output /home/users/yz1075/msdt/tmp_root_rashomon_frontier_electricity_d4.json
