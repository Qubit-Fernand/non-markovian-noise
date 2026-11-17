#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu3
# Mitigation error against circuit length, for static noise: for every number
# of Trotter steps the exact SPP (quadrature, no HEOM) and then the ZNE / PEC
# analysis of it.
#
#   bash scripts/scan_steps.sh                    # one Slurm job per number of steps
#   LOCAL=1 bash scripts/scan_steps.sh            # one after another, right here
#   STEPS_LIST="1 2 5 10 20" SIGMA=0.001 bash scripts/scan_steps.sh
#
# Progress, with elapsed time and an estimate of the time left:
#   tail -f results/steps_sig0.001_tb32/steps5*.log
# When the jobs have finished (finished ones are skipped when this is re-run):
#   python scripts/plot_steps.py results/steps_sig0.001_tb32     # table, steps.csv, steps.png
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "${BASH_SOURCE[0]}")/.."; fi
PYTHON="${PYTHON:-python}"
STEPS_LIST="${STEPS_LIST:-1 2 3 5 10 20}"   # Trotter steps (4 learning layers each)
EXAMPLE="${EXAMPLE:-ising4-sqisw}"
PULSE_ARGS="${PULSE_ARGS:---square-native}"
SIGMA="${SIGMA:-0.001}"                # static rms detuning per qubit, rad/ns
NODES="${NODES:-8}"                    # Gauss-Hermite nodes per qubit
TIME_BOND="${TIME_BOND:-32}"           # time bond the reference SPP is compressed to (16 loses probability)
CUTOFF="${CUTOFF:-1e-11}"
PEC_BONDS="${PEC_BONDS:-1,2,4,8}"      # time bond of the variational inverse
SWEEPS="${SWEEPS:-12}"
ROOT="${ROOT:-results/steps_sig${SIGMA}_tb${TIME_BOND}}"
CPUS="${CPUS:-8}"                      # cores (BLAS threads) per job
mkdir -p "$ROOT"

one() {
  local steps="$1" out="$ROOT/steps$1"
  if [[ ! -f "$out/summary.json" ]]; then
    if [[ -e "$out" ]]; then
      echo "UNFINISHED steps$steps: $out exists without summary.json (running, or failed: remove it to rerun)"; return
    fi
    echo "== steps $steps: reference SPP ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    "$PYTHON" -m heom_spp.cli reference --example "$EXAMPLE" $PULSE_ARGS --steps "$steps" \
      --bath static --sigma "$SIGMA" --nodes "$NODES" --time-bond "$TIME_BOND" \
      --compression-cutoff "$CUTOFF" --output "$out"
  fi
  if [[ ! -f "$out/mitigation.json" ]]; then
    echo "== steps $steps: ZNE / PEC analysis ($(date '+%F %T'))"
    "$PYTHON" -u -m heom_spp.analyze "$out" --pec-bonds "$PEC_BONDS" --sweeps "$SWEEPS"
  fi
  echo "== steps $steps finished ($(date '+%F %T'))"
}

if [[ -n "${ONLY_STEPS:-}" ]]; then        # inside a Slurm job
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}" \
         MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}"
  one "$ONLY_STEPS"
  exit
fi
for steps in $STEPS_LIST; do
  if [[ -f "$ROOT/steps$steps/mitigation.json" ]]; then
    echo "done      steps$steps"
  elif [[ -n "${LOCAL:-}" ]]; then
    export OMP_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS"
    one "$steps" 2>&1 | tee -a "$ROOT/steps$steps.log"
  else
    sbatch --job-name="steps$steps" --cpus-per-task="$CPUS" --output="$ROOT/steps$steps-%j.log" \
      --export=ALL,ONLY_STEPS="$steps" scripts/scan_steps.sh
  fi
done
echo
echo "Progress:  tail -f $ROOT/steps*.log"
echo "Then:      $PYTHON scripts/plot_steps.py $ROOT"
