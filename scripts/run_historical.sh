#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/stocks-historical-matplotlib"
exec .conda-env/bin/python -u -m stock_api.historical run --config "${1:?configuration path required}" --output "${2:?output path required}" --workers "${3:-1}" --report
