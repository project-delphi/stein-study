#!/usr/bin/env bash
# P2 handoff 1: tuning sweep (3 levels per arm x 2 validation seeds), then level selection.
# Idempotent: finished runs (runs/<hash>/result.json) are skipped. Prints PROGRESS lines.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
.venv/bin/python -m stein.sweep tune --workers 6 --out results/tuning.parquet
.venv/bin/python -m stein.select_tuning
echo "PROGRESS tuning done"
