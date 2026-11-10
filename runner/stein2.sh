#!/usr/bin/env bash
# Runner for the follow-up study (theory + frontier experiments). Same commands as stein.sh:
#
#   ./stein2.sh preflight | start | watch | status | stop
#
# It is stein.sh with a different spec and its own control directory, study2/ in the
# workspace (PROMPT.md, gates/, jobs/, logs/, state/), so the first study's gates, ledger
# and outputs are untouched and the spend counter starts at zero. Every setting can still
# be overridden from the environment, for example TOTAL_CAP=50 ./stein2.sh start.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STEIN_SUBDIR=study2
export STEIN_NAME="Stein study 2"
export STEIN_FINAL=FINAL2.md
export STEIN_PROMPT="${STEIN_PROMPT:-$HERE/stein2_prompt.md}"
export TOTAL_CAP="${TOTAL_CAP:-60}"                       # of the $70 left; the rest absorbs overshoot
export PHASE_BUDGETS="${PHASE_BUDGETS:-13 12 9 12 8}"     # P0 theory, P1 code, P2 sweep, P3 paper, P4 review
export COMPUTE_HOURS_P1="${COMPUTE_HOURS_P1:-1}"
export COMPUTE_HOURS_P2="${COMPUTE_HOURS_P2:-6}"
exec bash "$HERE/stein.sh" "$@"
