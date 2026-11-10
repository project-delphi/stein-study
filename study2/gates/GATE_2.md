# GATE 2 — Experiments and analysis (study 2) — IN PROGRESS

## Done
- Reviewed `study2/jobs/P2.sh` (drafted and smoke-tested in P1): `stein.sweep s2_all --workers 6`
  (1110 runs, blocks in priority order E1_core, E3_long, E2_linear, E1_alpha0.5, E1_d6, E1_alpha0.25,
  E1_d24, E4_noise, E5_ident, E6_covk; idempotent via `runs/<hash>/result.json`; PROGRESS after each
  finished run at most once a minute), then `stein.analyze2`. At start 1 of 1110 runs exists (the
  KSD λ 0.2 seed 1 rerun from validation B, shared by E1_core and E5).
- Handoff 1: `compute` written for the full grid (estimated 6.5 h at P1's per-arm times; the 6 h
  allowance will likely cut the tail, i.e. part of E5 and E6, in the prompt's cut order).

## Estimators (fixed before any held-out study-2 run was seen)
- Primary threshold: as in `theory/predictions.md` (smallest λ with 5-seed mean linear leakage R² > 0.10,
  log-λ interpolation, 2000-draw seed bootstrap with the same draw across λ; censored "<"/">").
  Secondary: nuisance variance share, τ = 0.05. Supplementary (GATE_1 D6): the same crossing searched
  upward from the λ of minimum mean leakage.
- Frontier: best 5-seed mean content R² over an arm's λ path subject to mean leakage ≤ ceiling
  (0.05, 0.10, 0.20), with seed-bootstrap CI; plus leakage at matched content R² (interpolated along
  the path) computed in the P2 analysis.

## Decisions
- D1. Threshold and frontier paths use only the registered λ grids (cells in blocks E1_*, E2_*, E6_*);
  `stein/analyze2.py` was edited so that E5 cells at the base setting with off-grid λ (KSD 0.04,
  SIGReg 0.05, MMD 2) do not add levels to the E1 paths. Made before any held-out result existed.

## Problems
- (none yet)

## Next
- After the job: run `.venv/bin/python -W ignore -m stein.analyze2` if the job was cut, read
  `results2/analysis.md`, write verdicts for P1–P18, answer the three questions, figures-ready tables.
