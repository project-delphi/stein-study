#!/bin/bash
# Study 2, P2 job (drafted in P1; P2 reviews it before handing off).
# Runs the study-2 grid (stein/grids.py s2_blocks, in priority order: E1 core, E3, E2 linear, E1 alpha 0.5,
# d 6, alpha 0.25, d 24, E4, E5, E6; all 5 seeds of a config together), then the analysis.
# Idempotent: finished runs (runs/<hash>/result.json) are skipped. stein.sweep prints PROGRESS every minute.
set -u
cd "$(dirname "$0")/../.."
.venv/bin/python -W ignore -m stein.sweep s2_all --workers 6 --out results2/sweep2.parquet
rc=$?
.venv/bin/python -W ignore -m stein.analyze2
exit $rc
