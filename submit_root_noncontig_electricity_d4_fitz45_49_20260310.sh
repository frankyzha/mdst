#!/usr/bin/env bash
#SBATCH --job-name=msdt_root_noncontig
#SBATCH --output=msdt_root_noncontig_%j.out
#SBATCH --error=msdt_root_noncontig_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=300G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]

set -euo pipefail

cd /home/users/yz1075/msdt
source .venv/bin/activate
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUT_JSON="/home/users/yz1075/msdt/tmp_root_noncontig_electricity_d4_${SLURM_JOB_ID:-manual}.json"

/usr/bin/time -v .venv/bin/python benchmark_root_rashomon_frontier.py \
  --seed 0 \
  --depth 4 \
  --lookahead 3 \
  --max-bins 255 \
  --min-child-size 8 \
  --proposal-atom-cap 32 \
  --max-branching 3 \
  --reg 0.0 \
  --root-time-limit 300 \
  --binner-estimators 2000 \
  --binner-early-stop 50 \
  --binner-threads "${SLURM_CPUS_PER_TASK:-6}" \
  --output "${OUT_JSON}"

.venv/bin/python - "${OUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))

def noncontiguous_group_count(group_spans):
    if not group_spans:
        return 0
    return sum(1 for group in group_spans if len(group) > 1)

strict_best = payload.get("strict_best")
frontier = payload.get("frontier_candidates", [])

strict_noncontig = noncontiguous_group_count(strict_best.get("group_spans")) if strict_best else 0
frontier_noncontig = [noncontiguous_group_count(c.get("group_spans")) for c in frontier]
frontier_best_noncontig = frontier_noncontig[0] if frontier_noncontig else 0
any_frontier_noncontig = any(v > 0 for v in frontier_noncontig)

summary = {
    "output_json": str(path),
    "best_exact_ub": payload.get("best_exact_ub"),
    "strict_best_feature": None if strict_best is None else strict_best.get("feature"),
    "strict_best_source": None if strict_best is None else strict_best.get("source"),
    "strict_best_noncontiguous_group_count": strict_noncontig,
    "frontier_size": len(frontier),
    "frontier_best_noncontiguous_group_count": frontier_best_noncontig,
    "any_frontier_noncontiguous": any_frontier_noncontig,
}
print(json.dumps(summary, indent=2, sort_keys=True))

if not any_frontier_noncontig:
    raise SystemExit("root frontier did not contain a non-contiguous partition")
PY
