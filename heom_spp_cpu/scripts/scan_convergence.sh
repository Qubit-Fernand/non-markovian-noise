#!/usr/bin/env bash
# Convergence scan of an SPP construction: one run per setting, one parameter
# changed at a time around a baseline (default: time bond 8, HEOM bond 8,
# 1 ns step, 6 hierarchy levels).
#
#   STEPS=1 bash scripts/scan_convergence.sh          # submit every run to Slurm
#   LOCAL=1 WORKERS=6 bash scripts/scan_convergence.sh  # run one after another here
#
# For static noise the exact SPP is first computed right here by quadrature
# (no HEOM, well under a minute per Trotter step, one core) into $ROOT/reference;
# `converge` then measures every HEOM run against it. REFERENCE=0 skips this.
#
# With a reference the scan need not move one parameter at a time. A grid of
# time step x HEOM bond (time bond and levels from TB, LV) is added by
#   GRID_DT="0.1 0.05" GRID_HB="8 16 24" bash scripts/scan_convergence.sh
# Runs are named dt<dt>-hb<hb>; runs already present are skipped.
#
# Independent-noise baselines for `heom_spp.analyze --baseline`: reset-layer and
# reset-pulse (ADOs zeroed after every learning layer / every pulse block).
# Submit them once the parameters are chosen, and nothing else:
#   BASELINES=only HB=16 DT=0.5 bash scripts/scan_convergence.sh
# (BASELINES=1 submits them together with the scan, at the baseline parameters.)
#
# When all runs have finished:
#   python -m heom_spp.cli converge results/scan_steps1
#
# Each run directory keeps its spp.npz, so the recommended one can go straight
# into `python -m heom_spp.analyze <run>` without recomputing anything.
# Re-running the script skips finished runs and reports unfinished directories.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
STEPS="${STEPS:-1}"                    # Trotter steps (4 learning layers each)
EXAMPLE="${EXAMPLE:-ising4-sqisw}"
PULSE_ARGS="${PULSE_ARGS:---square-native}"
SIGMA="${SIGMA:-0.001}"                # static rms detuning per qubit, rad/ns
TB="${TB:-8}";  TB_LIST="${TB_LIST:-16 32}"       # time bond of the SPP
HB="${HB:-8}";  HB_LIST="${HB_LIST:-16 24}"       # HEOM-MPS bond
DT="${DT:-1}";  DT_LIST="${DT_LIST:-0.5 0.25}"    # max time step, ns
LV="${LV:-6}";  LV_LIST="${LV_LIST:-8 10}"        # hierarchy levels per bath node
ROOT="${ROOT:-results/scan_steps${STEPS}}"
CPUS="${CPUS:-24}"                     # Slurm --cpus-per-task
WORKERS="${WORKERS:-4}"                # LOCAL=1 only
REFERENCE="${REFERENCE:-1}"
BASELINES="${BASELINES:-0}"
GRID_DT="${GRID_DT:-}"; GRID_HB="${GRID_HB:-}"
NODES="${NODES:-6}"                    # Gauss-Hermite nodes per qubit of the reference
mkdir -p "$ROOT"

if [[ "$REFERENCE" != 0 ]]; then
  if [[ -f "$ROOT/reference/summary.json" ]]; then
    echo "done      reference"
  elif [[ -e "$ROOT/reference" ]]; then
    echo "UNFINISHED reference: remove $ROOT/reference to rerun"
  else
    echo "running   reference (quadrature, local)"
    # shellcheck disable=SC2086
    "$PYTHON" -m heom_spp.cli reference --example "$EXAMPLE" $PULSE_ARGS --steps "$STEPS" \
      --bath static --sigma "$SIGMA" --nodes "$NODES" --output "$ROOT/reference" > "$ROOT/reference.log" 2>&1
    grep -E "quadrature_check_l2|correlation_l2" "$ROOT/reference.log" || true
  fi
fi

launch() {
  local name="$1" tb="$2" hb="$3" dt="$4" lv="$5" extra="${6:-}"
  local out="$ROOT/$name"
  if [[ -f "$out/summary.json" ]]; then
    echo "done      $name"; return
  fi
  if [[ -e "$out" ]]; then
    echo "UNFINISHED $name: $out exists without summary.json (still running, or failed: remove it to rerun)"; return
  fi
  local assembly=$(( tb * 4 > 32 ? tb * 4 : 32 ))
  local args="--example $EXAMPLE $PULSE_ARGS --steps $STEPS --bath static --sigma $SIGMA --levels $lv \
--engine tdvp --max-dt $dt --heom-bond $hb --time-bond $tb --assembly-bond $assembly $extra --output $out"
  if [[ -n "${LOCAL:-}" ]]; then
    echo "running   $name"
    # shellcheck disable=SC2086
    "$PYTHON" -m heom_spp.cli run $args --workers "$WORKERS" > "$ROOT/$name.log" 2>&1
  else
    RUN_ARGS="$args" sbatch --job-name="scan-$name" --cpus-per-task="$CPUS" \
      --output="$ROOT/$name-%j.log" scripts/spp.slurm
  fi
}

if [[ "$BASELINES" != only ]]; then
  launch base "$TB" "$HB" "$DT" "$LV"
  for v in $TB_LIST; do launch "tb$v" "$v" "$HB" "$DT" "$LV"; done
  for v in $HB_LIST; do launch "hb$v" "$TB" "$v" "$DT" "$LV"; done
  for v in $DT_LIST; do launch "dt$v" "$TB" "$HB" "$v" "$LV"; done
  for v in $LV_LIST; do launch "lv$v" "$TB" "$HB" "$DT" "$v"; done
  for dt in $GRID_DT; do for hb in $GRID_HB; do
    launch "dt$dt-hb$hb" "$TB" "$hb" "$dt" "$LV"
  done; done
fi
if [[ "$BASELINES" != 0 ]]; then
  launch reset-layer "$TB" "$HB" "$DT" "$LV" "--reset layer"
  launch reset-pulse "$TB" "$HB" "$DT" "$LV" "--reset pulse"
fi
echo
echo "When every run has finished:  $PYTHON -m heom_spp.cli converge $ROOT"
echo "Then, for the recommended run:  $PYTHON -m heom_spp.analyze $ROOT/<run> \\"
echo "    --baseline reset-layer=$ROOT/reset-layer --baseline reset-pulse=$ROOT/reset-pulse"
