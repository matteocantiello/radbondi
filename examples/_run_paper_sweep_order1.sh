#!/bin/bash
# Re-run just the order=1 masses with the fixed PAPER mode
# (convergence_tol=0.0, full 200k steps). See dispatch log for reasoning.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-/mnt/home/mcantiello/jupyter-env/bin/python3}
MAX_JOBS=${1:-11}
LOG_DIR="$REPO_ROOT/examples/paper_sweep_logs"
mkdir -p "$LOG_DIR"

# Order=1 masses only: the 5 N=6400 transitional + 6 N=3200 isothermal.
# These exited at step 5001 on the first pass because the residual dropped
# below the 1e-3 convergence tol before the flow had equilibrated.
MASSES=(
    -14.3 -14.0 -13.5 -13.3 -13.0        # order=1 N=6400 (5)
    -12.5 -12.0 -11.5 -11.0 -10.5 -10.0  # order=1 N=3200 (6)
)

echo "order=1 re-run: ${#MASSES[@]} masses, up to $MAX_JOBS in parallel" \
     "(start $(date -Iseconds))" | tee -a "$LOG_DIR/_dispatch.log"

run_one() {
    local logm="$1"
    local tag
    tag=$(printf 'logM%+06.2f' "$logm")
    local log="$LOG_DIR/${tag}.log"
    local t0 t1 status
    t0=$(date +%s)
    echo "[$(date -Iseconds)] start $logm -> $log"
    if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MPLBACKEND=Agg \
       RADBONDI_PAPER=1 RADBONDI_LOGM="$logm" \
       "$PY" "$REPO_ROOT/examples/02_paper_sweep.py" > "$log" 2>&1; then
        status=OK
    else
        status=FAIL
    fi
    t1=$(date +%s)
    echo "[$(date -Iseconds)] done  $logm   $status  dt=$((t1-t0))s"
}

export -f run_one
export PY REPO_ROOT LOG_DIR

printf '%s\n' "${MASSES[@]}" | \
    xargs -n 1 -P "$MAX_JOBS" -I {} bash -c 'run_one "$@"' _ {} \
    | tee -a "$LOG_DIR/_dispatch.log"

echo "order=1 re-run: finished $(date -Iseconds)" \
     | tee -a "$LOG_DIR/_dispatch.log"
