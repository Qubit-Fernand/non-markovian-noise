#!/usr/bin/env bash
# First establish correctness with smoke.sh. No SSH or job submission is performed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/scaling_$(date +%Y%m%d_%H%M%S)}"
WORKERS_LIST="${WORKERS_LIST:-1 2 4 8 12 16 24 32 48}"
ENGINES="${ENGINES:-tdvp}"
REPEATS="${REPEATS:-3}"
BLAS_THREADS="${BLAS_THREADS:-1}"
export OMP_NUM_THREADS="$BLAS_THREADS" OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS" BLIS_NUM_THREADS="$BLAS_THREADS"
mkdir -p "$OUT_ROOT"
"$PYTHON" -m pip freeze > "$OUT_ROOT/pip-freeze.txt"
"$PYTHON" -c 'import os,platform; print(platform.platform()); print("affinity",sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count())' > "$OUT_ROOT/hardware.txt"
if command -v lscpu >/dev/null; then lscpu >> "$OUT_ROOT/hardware.txt"; fi
limit="$("$PYTHON" -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count())')"
# Minimal baseline: HTML four-qubit circuit, independent static coordinate on EVERY qubit.
# levels is the dimension of each bath node, not the number of bath nodes.
common=(--example ising4 --bath static --coupled-qubits all --modes-per-qubit 1
        --levels 3 --heom-bond 8 --time-bond 8 --assembly-bond 32
        --max-dt 2 --blas-threads "$BLAS_THREADS" --chunk-size 8)
for engine in $ENGINES; do
  for workers in $WORKERS_LIST; do
    if (( workers * BLAS_THREADS > limit )); then
      printf 'Skipping workers=%s: allocation is %s CPUs\n' "$workers" "$limit"
      continue
    fi
    for (( repeat=1; repeat<=REPEATS; repeat++ )); do
      tag="${engine}_w${workers}_t${BLAS_THREADS}_r${repeat}"
      command=("$PYTHON" -m heom_spp.cli run "${common[@]}" "$@" \
               --engine "$engine" --workers "$workers" --output "$OUT_ROOT/$tag")
      if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "$OUT_ROOT/$tag.time.txt" "${command[@]}" | tee "$OUT_ROOT/$tag.log"
      else
        "${command[@]}" | tee "$OUT_ROOT/$tag.log"
      fi
      baseline="$OUT_ROOT/${engine}_w1_t${BLAS_THREADS}_r1"
      if [[ -f "$baseline/summary.json" && "$tag" != "${engine}_w1_t${BLAS_THREADS}_r1" ]]; then
        "$PYTHON" -m heom_spp.cli compare "$baseline" "$OUT_ROOT/$tag" \
          --tolerance 1e-9 --output "$OUT_ROOT/${tag}_compare.json"
      fi
    done
  done
done
"$PYTHON" -m heom_spp.cli summarize "$OUT_ROOT" --output "$OUT_ROOT/scaling.json"
