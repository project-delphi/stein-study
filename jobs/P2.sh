#!/usr/bin/env bash
# P2 handoff 2: held-out sweep (5 seeds {1..5}) at the levels in results/tuning_selected.json
# (written by handoff 1: `python -m stein.sweep tune` + `python -m stein.select_tuning`).
# Blocks in priority order H1, H4, H2, H3, H5 (temporal), CIFAR, H6 (VAE).
# Idempotent: finished runs (runs/<hash>/result.json) are skipped. Prints PROGRESS lines.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
test -f results/tuning_selected.json
.venv/bin/python -m stein.sweep heldout --workers 6 --out results/heldout.parquet
echo "PROGRESS held-out done"
