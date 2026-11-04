# Four-qubit Ising implementation check — 2026-11-04

Four finite-control slots, one OU mode on qubit 0, three hierarchy levels. Compressed RK4 max_dt=0.1 is compared to the same-depth sparse-matrix exponential. Neither run enumerates 256^4 histories.

- Joint-tensor L2 difference, contracted as a tensor network: 2.4683933e-05.
- Largest single-slot marginal difference: 1.0237605e-05.
- MPS elapsed: 213.087 s; dense-reference elapsed: 10.249 s.
- MPS time bonds: (3, 3, 3); dense-reference time bonds: (2, 2, 2).
- MPS normalization error: 1.7763569e-15.

The prototype MPS propagator is substantially slower here. These runs were launched concurrently and are implementation checks, not controlled performance benchmarks. The different retained ranks reflect the absolute SVD cutoff and propagation error. This low-depth, single-mode check does not establish multimode physical convergence or a scaling advantage. Full-history positivity was not checked; no sampling or PEC accuracy claim is made for this four-qubit tensor.

See `comparison.json`, `summary.json`, and the sibling dense-reference directory. Source snapshots and parameter files are included in each run.
