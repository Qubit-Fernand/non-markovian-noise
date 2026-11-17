#!/usr/bin/env bash
#SBATCH --job-name=heom-spp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu3
#SBATCH --output=heom-spp-%j.log
# CPU-only job on a GPU-partition node (gpu-node2/3/4: 52 cores, ~385 GB each).
# No GPU is requested. If the partition refuses jobs without one, add
#   #SBATCH --gres=gpu:tesla:1
# Submit from the repository root with the Python environment already active:
#   sbatch scripts/spp.slurm                       # scaling scan (default)
#   sbatch --cpus-per-task=48 scripts/spp.slurm    # more cores, if a node is free
#   RUN_ARGS="--layers-json configs/html_ising4_layers.json --noise-json configs/static4_noise.json \
#             --output results/static4" sbatch scripts/spp.slurm   # one production run
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
CPUS="${SLURM_CPUS_PER_TASK:?}"
export PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
if [[ -n "${RUN_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  srun --cpu-bind=cores "$PYTHON" -m heom_spp.cli run $RUN_ARGS --workers "$CPUS" --blas-threads 1
else
  export WORKERS_LIST="${WORKERS_LIST:-1 2 4 8 12 16 24 32 48}"   # entries above $CPUS are skipped
  export REPEATS="${REPEATS:-3}"
  export OUT_ROOT="${OUT_ROOT:-results/scaling_${SLURM_JOB_ID}}"
  srun --cpu-bind=cores bash scripts/scaling.sh
fi
