#!/bin/bash
#SBATCH --job-name=check_perf
#SBATCH --partition=compsci-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --nodelist=linux52
#SBATCH --output=results/runtime_compare_jobs/check_perf_%j.out
#SBATCH --error=results/runtime_compare_jobs/check_perf_%j.err
set -euo pipefail
which perf || true
perf --version || true
