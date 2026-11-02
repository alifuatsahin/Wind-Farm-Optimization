#!/usr/bin/env bash
#
# Run the optimiser-efficiency N_t sweep (fig:bo_efficiency panel b) on a Linux box.
#
# The sweep is tens of thousands of farm evaluations and costs hours, so this script
# refuses to start one until it has checked three things that each silently waste the
# whole run if they are wrong:
#
#   THE BUILD COMPUTES THE RIGHT ANSWER. Two reference objectives are re-evaluated and
#   compared against values measured on the development machine. A GPU jaxlib is a
#   DIFFERENT build with different kernels and a different default matmul precision, and
#   nothing else in the run would notice it quietly returning something else.
#
#   THE HARDWARE IS ACTUALLY FASTER. The model has never been benchmarked on a GPU. The
#   march is thousands of sequential small steps, which is the shape of problem a GPU is
#   WORST at, so a GPU run losing to the CPU is a live possibility and not a surprise.
#   One evaluation is timed at each end of the sweep and the total is extrapolated before
#   anything is committed.
#
#   THE COST IS WHAT YOU EXPECT. The estimate is printed and confirmed before the sweep
#   starts. Pass --yes to skip the prompt for an unattended run.
#
# The sweep is RESUMABLE. Every (method, N_t, seed) curve is written to
# runs/paper_figures/bo_efficiency_runs.json as soon as it finishes, and a re-run reads
# them back, so a killed or preempted job loses at most the curve in flight. Copy that
# file back and forth to move progress between machines.
#
# Usage:
#   ./run_sweep.sh --bundle sweep.tar.gz  # ON THIS MACHINE: pack what the box needs
#   ./run_sweep.sh --install --gpu        # set up a venv with CUDA jax, then sweep
#   ./run_sweep.sh --check-only           # verify and benchmark, run nothing
#   ./run_sweep.sh --nt "2 3 4 6" --seeds-b 3 --yes
#
# A GIT CLONE IS NOT ENOUGH. Four of the files this sweep runs are in .gitignore --
# make_paper_figures.py, baselines.py, validation.py and validation_fullwake.py -- because
# the OTHER things in them read the non-redistributable PIV and LES data. The sweep
# itself reads none of it (verified: importing all four opens nothing under data/), so
# --bundle packs the tracked repo plus exactly those four and nothing else. Without it
# the box gets a clone that cannot run the figure at all.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NT="2 3 4 6 8"
SEEDS=8
SEEDS_B=5
BUDGET=120
EVALS_PER_DIM=50
DO_INSTALL=0
USE_GPU=0
CHECK_ONLY=0
ASSUME_YES=0
VENV="$REPO/.venv"
PYTHON="${PYTHON:-python3}"
BUNDLE=""

# Gitignored, and the sweep cannot run without them. tools/sweep_precheck.py is listed
# too: it is new, so a clone made before it was committed will not have it either.
REQUIRED="make_paper_figures.py baselines.py validation.py validation_fullwake.py tools/sweep_precheck.py"

while [ $# -gt 0 ]; do
  case "$1" in
    --bundle)        BUNDLE="$2"; shift 2 ;;
    --nt)            NT="$2"; shift 2 ;;
    --seeds)         SEEDS="$2"; shift 2 ;;
    --seeds-b)       SEEDS_B="$2"; shift 2 ;;
    --budget)        BUDGET="$2"; shift 2 ;;
    --evals-per-dim) EVALS_PER_DIM="$2"; shift 2 ;;
    --venv)          VENV="$2"; shift 2 ;;
    --python)        PYTHON="$2"; shift 2 ;;
    --install)       DO_INSTALL=1; shift ;;
    --gpu)           USE_GPU=1; shift ;;
    --check-only)    CHECK_ONLY=1; shift ;;
    --yes|-y)        ASSUME_YES=1; shift ;;
    -h|--help)       sed -n '2,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }

# ----------------------------------------------------------------- 0. files
if [ -n "$BUNDLE" ]; then
  say "packing $BUNDLE"
  missing=""
  for f in $REQUIRED; do [ -f "$f" ] || missing="$missing $f"; done
  if [ -n "$missing" ]; then
    echo "cannot bundle, missing:$missing" >&2; exit 1
  fi
  # git ls-files gives the tracked repo without runs/, data/ or any local junk; the
  # REQUIRED four are appended because git deliberately does not know about them.
  # The run cache rides along when it exists: runs/ is gitignored, but those curves are
  # hours of CPU each and the sweep reads them back instead of repeating them.
  cache="runs/paper_figures/bo_efficiency_runs.json"
  { git ls-files; printf '%s\n' $REQUIRED
    [ -f "$cache" ] && printf '%s\n' "$cache"; } | sort -u > /tmp/sweep_files.txt
  # --force-local when tar has it: bundling is normally done on the DEVELOPMENT machine,
  # which is Windows, and there a destination like C:/tmp/x.tar.gz is read as the remote
  # host "C" and the bundle fails with "cannot connect".
  tarlocal=""
  tar --help 2>/dev/null | grep -q -- --force-local && tarlocal="--force-local"
  # shellcheck disable=SC2086
  tar $tarlocal -czf "$BUNDLE" -T /tmp/sweep_files.txt
  echo "  $(wc -l < /tmp/sweep_files.txt) files, $(du -h "$BUNDLE" | cut -f1)"
  echo "  no data/ -- the sweep reads none of it"
  [ -f "$cache" ] && echo "  includes the run cache, so the box resumes instead of restarting"
  echo
  echo "  scp $BUNDLE user@box:~/ && ssh user@box"
  echo "  mkdir -p wfo && tar xzf $(basename "$BUNDLE") -C wfo && cd wfo"
  echo "  ./run_sweep.sh --install --gpu --check-only"
  exit 0
fi

missing=""
for f in $REQUIRED; do [ -f "$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "MISSING (gitignored -- a clone will not have them):$missing" >&2
  echo "Run '--bundle sweep.tar.gz' on the development machine and copy that here." >&2
  exit 1
fi

# ----------------------------------------------------------------- 1. environment
if [ "$DO_INSTALL" = 1 ]; then
  say "creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install --upgrade pip wheel >/dev/null
  if [ "$USE_GPU" = 1 ]; then
    # The CUDA wheels bundle their own runtime, so no system CUDA install is needed --
    # only a driver new enough for CUDA 12.
    say "installing jax with CUDA 12 support"
    pip install "jax[cuda12]>=0.11"
  else
    say "installing CPU jax"
    pip install "jax>=0.11" "jaxlib>=0.11"
  fi
  # requirements.txt pins jax too; install the rest without letting it overwrite the
  # CUDA jaxlib that was just chosen.
  grep -vE '^[[:space:]]*(jax|jaxlib)\b' requirements.txt > /tmp/req_nojax.txt
  pip install -r /tmp/req_nojax.txt
elif [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
fi

# fp32 is the default and is what the GPU needs: a consumer card runs fp64 at 1/64 of
# fp32, so a double-precision GPU run can lose to the CPU outright. Set here explicitly
# rather than relied on, because the reference values below were measured in fp32.
export WFO_FLOAT64=0
export MPLBACKEND=Agg
# JAX grabs 75% of GPU memory on first use by default, which collides with anything else
# on a shared box and buys nothing here -- the arrays are small.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

say "environment"
python - <<'PY'
import jax, platform
print("  python   %s" % platform.python_version())
print("  jax      %s" % jax.__version__)
print("  backend  %s" % jax.default_backend())
for d in jax.devices():
    print("  device   %r" % (d,))
PY

# ----------------------------------------------------------------- 2. correctness
say "verifying the build against reference objectives"
python "$REPO/tools/sweep_precheck.py" --verify || {
  echo "VERIFICATION FAILED -- not starting a multi-hour sweep on this build." >&2
  exit 1
}

# ----------------------------------------------------------------- 3. cost
say "timing one evaluation at each end of the sweep"
python "$REPO/tools/sweep_precheck.py" --benchmark \
    --nt "$NT" --seeds-b "$SEEDS_B" --budget "$BUDGET" \
    --evals-per-dim "$EVALS_PER_DIM"

if [ "$CHECK_ONLY" = 1 ]; then
  say "--check-only: stopping before the sweep"
  exit 0
fi

if [ "$ASSUME_YES" != 1 ]; then
  printf '\nStart the sweep? [y/N] '
  # `|| reply=` so that EOF on a non-interactive stdin aborts cleanly instead of being
  # killed by set -e, which would look like a crash rather than a declined prompt.
  read -r reply || reply=""
  case "$reply" in [yY]*) ;; *) echo "aborted (pass --yes for an unattended run)"; exit 0 ;; esac
fi

# ----------------------------------------------------------------- 4. sweep
LOG="$REPO/runs/paper_figures/sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
say "running; log -> $LOG"
echo "  (safe to kill and re-run: finished curves are cached in bo_efficiency_runs.json)"

# set +e around the pipeline: with pipefail, a failing sweep would otherwise be killed
# by set -e right here and never print where its partial results landed -- which is the
# one moment the paths matter most.
set +e
# shellcheck disable=SC2086
python -u make_paper_figures.py --only bo_efficiency \
    --nt $NT --seeds "$SEEDS" --seeds-b "$SEEDS_B" \
    --budget "$BUDGET" --evals-per-dim "$EVALS_PER_DIM" 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
set -e
say "sweep exited with status $status"
echo "  figure   runs/paper_figures/bo_efficiency.png"
echo "  summary  runs/paper_figures/bo_efficiency_summary.json"
echo "  raw      runs/paper_figures/bo_efficiency_runs.json   <- copy this back"
exit "$status"
