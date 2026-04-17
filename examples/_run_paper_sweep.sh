#!/bin/bash
# Launch the full PAPER_CONFIG mass sweep in parallel.
# Each mass runs in its own Python process with a dedicated log file.
# BLAS threading is pinned to 1 per process so the 18 runs don't oversubscribe.
#
# Usage:
#   examples/_run_paper_sweep.sh [MAX_JOBS]
#
# Default MAX_JOBS is 14 (leaves headroom on a 16-core box).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-/mnt/home/mcantiello/jupyter-env/bin/python3}
MAX_JOBS=${1:-14}
LOG_DIR="$REPO_ROOT/examples/paper_sweep_logs"
mkdir -p "$LOG_DIR"

# All 18 masses from PAPER_CONFIG, ordered by estimated cost
# (heaviest first so they start earliest and set the wall-time floor).
# MUSCL N=6400 >> order=1 N=6400 >> order=1 N=3200.
MASSES=(
    -16.1 -16.0 -15.6 -15.3 -15.1 -15.0 -14.52   # MUSCL N=6400 (7)
    -14.3 -14.0 -13.5 -13.3 -13.0                # order=1 N=6400 (5)
    -12.5 -12.0 -11.5 -11.0 -10.5 -10.0          # order=1 N=3200 (6)
)

echo "paper sweep: ${#MASSES[@]} masses, up to $MAX_JOBS in parallel" \
     "(start $(date -Iseconds))" | tee "$LOG_DIR/_dispatch.log"

run_one() {
    local logm="$1"
    local tag
    tag=$(printf 'logM%+06.2f' "$logm")
    local log="$LOG_DIR/${tag}.log"
    local t0 t1 status
    t0=$(date +%s)
    echo "[$(date -Iseconds)] start $logm -> $log"
    # Pin BLAS threads so parallel masses don't fight each other.
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

# xargs -P runs up to MAX_JOBS masses in parallel. Each invocation of
# run_one handles a single mass; output is captured in per-mass log files.
printf '%s\n' "${MASSES[@]}" | \
    xargs -n 1 -P "$MAX_JOBS" -I {} bash -c 'run_one "$@"' _ {} \
    | tee -a "$LOG_DIR/_dispatch.log"

echo "paper sweep: all dispatched, finished $(date -Iseconds)" \
     | tee -a "$LOG_DIR/_dispatch.log"
