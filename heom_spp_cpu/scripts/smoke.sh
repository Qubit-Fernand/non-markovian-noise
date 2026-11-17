#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/smoke_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p "$OUT_ROOT"
"$PYTHON" -m pytest tests/test_heom.py tests/test_circuit_bath.py -q
common=(--example one-qubit --bath static --modes-per-qubit 1 --levels 6
        --rotation-ns 1 --max-dt 0.1 --sigma 0.1
        --heom-bond 6 --time-bond 6 --assembly-bond 32 --chunk-size 2)
for engine in dense rk4 tdvp; do
  "$PYTHON" -m heom_spp.cli run "${common[@]}" --engine "$engine" \
    --workers 1 --output "$OUT_ROOT/${engine}_w1"
done
"$PYTHON" -m heom_spp.cli run "${common[@]}" --engine tdvp \
  --workers 2 --output "$OUT_ROOT/tdvp_w2"
for engine in rk4 tdvp; do
  "$PYTHON" -m heom_spp.cli compare "$OUT_ROOT/dense_w1" "$OUT_ROOT/${engine}_w1" \
    --tolerance 2e-6 --output "$OUT_ROOT/${engine}_accuracy.json"
done
"$PYTHON" -m heom_spp.cli compare "$OUT_ROOT/tdvp_w1" "$OUT_ROOT/tdvp_w2" \
  --tolerance 1e-10 --output "$OUT_ROOT/parallel_accuracy.json"
