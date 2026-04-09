#!/bin/bash
#SBATCH --job-name=cmp_cached_msplit_heloc_l4_r5e-4
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/runtime_compare_jobs/cmp_cached_msplit_heloc_l4_r5e-4_%j.out
#SBATCH --error=results/runtime_compare_jobs/cmp_cached_msplit_heloc_l4_r5e-4_%j.err

set -euo pipefail

cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs

RUN_NAME="cached_msplit_linear_nonlinear_shapecart_heloc_lookahead4_reg0005_$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="results/comparison_cached_msplit_linear_nonlinear_shapecart_heloc_lookahead4_reg0005"
PROFILE_PREFIX="results/runtime_compare_jobs/${RUN_NAME}_${SLURM_JOB_ID}"

CMD=(
  .venv/bin/python
  run_cached_depth_benchmarks_msplit_linear_nonlinear_shapecart.py
  --datasets heloc
  --depths 2 3 4 5 6
  --seed 0
  --test-size 0.2
  --val-size 0.1
  --max-bins 1024
  --min-samples-leaf 4
  --min-child-size 4
  --lookahead-depth 4
  --reg 0.0005
  --max-branching 3
  --lgb-num-threads 6
  --shape-k 3
  --shape-min-samples-leaf 4
  --shape-min-samples-split 8
  --shape-inner-max-depth 6
  --shape-inner-max-leaf-nodes 24
  --shape-max-iter 10
  --msplit-min-child-size 4
  --msplit-min-split-size 8
  --msplit-family1-soft-weight 1.0
  --results-root "${RESULTS_ROOT}"
  --run-name "${RUN_NAME}"
)

/usr/bin/time -v -o "${PROFILE_PREFIX}.time.log" "${CMD[@]}"
printf '%q ' "${CMD[@]}" > "${PROFILE_PREFIX}.cmd.txt"
printf '\n' >> "${PROFILE_PREFIX}.cmd.txt"

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=$(realpath "${RESULTS_ROOT}/${RUN_NAME}")"
echo "TIME_LOG=$(realpath "${PROFILE_PREFIX}.time.log")"
echo "CMD_LOG=$(realpath "${PROFILE_PREFIX}.cmd.txt")"
