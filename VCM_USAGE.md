# Running This Project on Duke VCM

## Why this workflow
Duke VCM documentation highlights that VM/container sessions can be reset and data can be lost if it is not backed up. This project writes every run to a dedicated folder, checkpoints per-trial CSV rows, and creates a single artifact bundle (`.tar.gz`) for easy download/copy.

## One-command run
From project root:

```bash
./run_vcm_multisplit.sh
```

Optional example with custom settings:

```bash
RUN_NAME=my_vcm_run ./run_vcm_multisplit.sh \
  --datasets electricity eye-movements eye-state \
  --depth-budgets 2 3 4 5 6 \
  --seeds 0 1 2 3 4 \
  --time-limit 120 \
  --resume
```

## Where outputs go
Each run is written to:

```text
results/runs/<run_name>/
```

Key files:
- `summary_results.csv`
- `seed_results.csv`
- `multisplit_depth_vs_accuracy.log`
- `multisplit_cart_dp_accuracy.png`
- `multisplit_cart_dp_vs_paper_accuracy.png`
- `<run_name>_artifacts.tar.gz`
- `manifest.json`

Compatibility copies are also written to:

```text
results/multisplit_cart_dp_results.csv
results/multisplit_cart_dp_accuracy.png
results/multisplit_depth_vs_accuracy.log
```

## Retrieval tips
Use the bundle file from `results/runs/<run_name>/<run_name>_artifacts.tar.gz` as the primary download artifact.

If you mount or sync to persistent storage (for example CIFS/Box/Git as noted by Duke OIT), you can also run:

```bash
./run_vcm_multisplit.sh --copy-to /path/to/persistent/storage
```

## Resume interrupted runs
If a session is interrupted, rerun with the same `RUN_NAME` and `--resume`.

```bash
RUN_NAME=my_vcm_run ./run_vcm_multisplit.sh --resume
```

Completed `(dataset, depth, seed)` trials are skipped.

## LightGBM preprocessing variant
Install dependency once in your active environment:

```bash
.venv/bin/pip install lightgbm
```

On macOS, if the prebuilt wheel fails because of OpenMP / `libomp`, build without OpenMP:

```bash
.venv/bin/pip uninstall -y lightgbm
CMAKE_ARGS='-DUSE_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES=arm64' \
  .venv/bin/pip install --no-binary :all: --no-cache-dir lightgbm
```

To run the LightGBM-threshold preprocessing variant (separate from CART):

```bash
./run_vcm_multisplit_lightgbm.sh
```

This writes run folders under:

```text
results/runs_lightgbm/<run_name>/
```

and stable compatibility files under:

```text
results/lightgbm/multisplit_lightgbm_dp_results.csv
results/lightgbm/multisplit_lightgbm_dp_accuracy.png
results/lightgbm/multisplit_lightgbm_depth_vs_accuracy.log
```
