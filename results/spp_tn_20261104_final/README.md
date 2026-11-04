# SPP tensor-network synthetic benchmark — 2026-11-04

Run: `python -m spp_tn.demo --output results/spp_tn_20261104_final --convergence`.

One qubit, three finite-control slots, one classical OU mode; gate-after errors. This is an exact-model synthetic benchmark, with no fitting or hardware measurements.

- Probability normalization error: 3.55e-15.
- MPS vs same-depth sparse-exponential reference, probability L2: 1.93923e-08.
- Variational vs dense PEC inverse L2: 1.37724e-12.
- PEC Gamma squared: 1.47297145.

| Response | Expectation |
|---|---:|
| ideal | 0.961055438311 |
| noisy | 0.794723643390 |
| independent_prediction | 0.778382991126 |
| independent_pec | 0.981230946900 |
| correlated_pec | 0.961055438309 |
| pea_quadratic_extrapolation | 0.956073149015 |

Finite-shot PEC: 0.961462 ± 0.005237 (one standard error, 20,000 shots; no learning uncertainty).

| Convergence axis | Value | Probability L2 error |
|---|---:|---:|
| hierarchy_levels | 3 | 0.00082236 |
| hierarchy_levels | 5 | 3.46743e-06 |
| hierarchy_levels | 7 | 7.04349e-09 |
| rk4_max_dt | 0.2 | 2.66137e-06 |
| rk4_max_dt | 0.1 | 3.00534e-07 |
| rk4_max_dt | 0.05 | 1.93923e-08 |
| internal_heom_bond | 1 | 0.0959024 |
| time_bond | 1 | 0.00297483 |
| internal_heom_bond | 2 | 0.00641506 |
| time_bond | 2 | 1.33334e-05 |
| internal_heom_bond | 4 | 1.93923e-08 |
| time_bond | 4 | 0 |

Hierarchy-level scans use levels=10 as reference. RK4 and bond scans use the same levels=5 sparse-exponential reference. Strong truncations may violate positivity/normalization: inspect their full probability reports in `summary.json`; no repair was applied.

`source/` and `snapshot_sha256` contain the source used for this run. `samples.npz` retains noise histories, corrections, weights, and measurement outcomes. The snapshot includes the tests present at run time; three additional analytic OU/Markov tests were subsequently added to the working tree.
