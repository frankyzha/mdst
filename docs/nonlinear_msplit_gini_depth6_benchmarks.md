# Nonlinear MSPLIT Gini-Only Benchmark Snapshot

This note records the fixed-configuration benchmark used to validate the contiguous-DP acceleration changes in nonlinear MSPLIT.

## Configuration

- depth: `6`
- lookahead depth: `3`
- regularization: `0.0`
- min split size: `8`
- min child size: `4`
- max branching: `3`
- exactify top-k: `2`
- worker limit: `1`

All runs use the cached LightGBM-bin splits under `benchmark/cache/`.

## Baseline

| Dataset | Fit time (s) | Test accuracy |
| --- | ---: | ---: |
| `compas` | `0.0381` | `0.646165` |
| `coupon` | `0.2226` | `0.682119` |
| `electricity` | `12.9633` | `0.883482` |
| `eye-movement` | `4.8098` | `0.639947` |
| `eye-state` | `2.0295` | `0.786048` |
| `heloc` | `0.9480` | `0.676000` |
| `spambase` | `0.7409` | `0.905537` |

## Optimized

| Dataset | Fit time (s) | Test accuracy |
| --- | ---: | ---: |
| `compas` | `0.0343` | `0.646165` |
| `coupon` | `0.2071` | `0.682119` |
| `electricity` | `11.8909` | `0.883482` |
| `eye-movement` | `4.7486` | `0.639947` |
| `eye-state` | `1.9291` | `0.786048` |
| `heloc` | `0.9515` | `0.676000` |
| `spambase` | `0.7036` | `0.905537` |

## Delta

| Dataset | Speedup | Test-accuracy delta |
| --- | ---: | ---: |
| `compas` | `1.1112x` | `0.000000` |
| `coupon` | `1.0746x` | `0.000000` |
| `electricity` | `1.0902x` | `0.000000` |
| `eye-movement` | `1.0129x` | `0.000000` |
| `eye-state` | `1.0520x` | `0.000000` |
| `heloc` | `0.9963x` | `0.000000` |
| `spambase` | `1.0530x` | `0.000000` |

The exact optimization changes preserved both the objective value and the test accuracy on all seven cached datasets in this sweep.
