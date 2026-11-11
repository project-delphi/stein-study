# GATE 1 — Implementation and validation (study 2) — FINAL

## Done
- **Code (cache-safe).** New config keys in `stein/train.py` `HASH_OPTIONAL`, each omitted from `cfg_hash`
  at its default, each default = study 1's behaviour: `encoder` (conv | linear), `pred_arch` (mlp | linear),
  `noise_mode` (batch | fixed), `eval_v` (1 | 2). New arms `cov` (‖Σ̂ − I‖²_F, GATE_0 D4) and `covk` (E6) in
  `stein/losses.py`; tier `synth` (linear-Gaussian views, k = 6, m = 5) in `stein/data.py`;
  `LinearEncoder`, `LinearPredictor` in `stein/models.py`; metrics in `stein/metrics.py`
  (`study2_metrics`: content/nuisance/joint/unexplained variance shares, linear-code and one-hot;
  explained-covariance eigenvalues `nuis_eigs`/`content_eigs`; principal-axis decomposition
  `pc_var`/`pc_nuis_var`/`pc_content_var`; `nuis_dir_ratio`; `mcc`, `mcc_per`).
  Grids in `stein/grids.py` (`s2_pilot`, `s2_blocks`, `s2_all`, `s2_block`, `s2_tiny`, `s2_membership`);
  analysis `stein/analyze2.py` (→ `results2/runs2.parquet`, `cells2.parquet`, `thresholds.parquet`,
  `frontier.parquet`, `schema.json`, `analysis.md`); validation `stein/validate2.py`
  (→ `results2/validation2.json`); P2 job draft `study2/jobs/P2.sh`.
- **Validation gates — all pass.**
  | gate | result |
  |---|---|
  | existing 28 tests | pass |
  | cache-hash test (`test_cached_run_hashes_unchanged`): every cached run re-hashes as under study 1's code; the 817 runs with complete configs re-hash to their directory | pass (27 early study-1 pilot runs predate some study-1 keys and were already orphaned in study 1; they hash identically under old and new code) |
  | MCC = 1 for a scaled signed permutation; < 0.9 for a random rotation; 1/√2 for a 45° pair | pass |
  | variance shares / eigenvalues / principal axes on a constructed mixture (6/14, 5/14, 3/14) | pass |
  | one-hot share sees cos 2h that the linear code misses | pass |
  | `cov` ≈ finite-N bias (d+d²)/(N−1) on N(0, I); = formula on rank-6 collapse; large on anisotropic; gradcheck (cov, covk) | pass |
  | `covk` ignores empty spare directions | pass |
  | fixed vs batch noise scale | pass |
  | λ̂* estimator (log interpolation, censoring, upward variant) and frontier on fake paths | pass |
  | **theory meets code** (`validate2.py` A: real loop, `synth` + `linear` encoder + `linear` predictor + `cov`, α = 1, d = 12, 4000 steps, λ = q·λ*, λ* = 1/(2d), q ∈ {0.5, 0.8, 1.25, 2, 5}, seeds 0–1) | **pass**: onset between 0.8λ* and 1.25λ* in both seeds (v < 0.01 below, > 0.1 above); mean v and c within 0.03 of the finite-N reference in 9/10 runs (q = 2: 0.467/0.469 vs 0.466; q = 5: 0.761/0.764 vs 0.760); the 10th (q = 1.25, seed 0) has 4 active nuisance directions at mean 0.176 vs 0.173 and one trapped empty (0.001, still at 12 000 steps) |
  | rerun of a cached study-1 config (KSD, λ 0.2, seed 1, base) under `eval_v=2` | **pass, exact**: all 11 shared metrics identical (MPS training deterministic) |
  Total: 40 tests pass (`.venv/bin/python -m pytest -q tests`).
- **Pilot** (seed 0, 30 runs, `results2/pilot_p1.parquet`; never reported). Train+eval time per run
  under 6 concurrent MPS workers: cov 73 s, covk ~30 s (ran with little contention), KSD 115, SIGReg 130,
  MMD 135, sliced KSD 182, VICReg 101, linear-encoder cov 95, linear-encoder KSD 78; effective
  **18 s/run** with 6 workers. Linear leakage R² on the λ scan (with cached held-out means for orientation):
  cov 0.01→0.28, 0.02→0.27, 0.04→0.15, 0.08→0.00, 0.16→0.14, 0.32→0.37; SIGReg 0.005→0.005, 0.02→0.001,
  0.1→0.14; KSD 0.1→0.003, 0.4→0.34, 2→0.58; sliced KSD 0.5→0.34, 2.5→0.009, 10→0.49; MMD 5→0.00,
  20→0.13, 100→0.12; VICReg 0.05/0.2/0.5 → 0.10/0.08/0.15 (content R² ≈ 0.2, collapse regime);
  linear-encoder cov 0.01/0.04/0.16 → 0.41/0.002/0.031, linear-encoder KSD 0.2/1/5 → 0.014/0.037/0.42.
  Rough onsets: cov ~0.1–0.16, SIGReg ~0.04, KSD ~0.2–0.4, sliced KSD ~3–5, MMD ~10–20, VICReg < 1
  (cached: 1→0.18, 5→0.72); linear encoder: cov > 0.16, KSD 1–5.
- **λ grids for P2** (`LAM2`, 7 levels, factor 2): cov 0.02–1.28; SIGReg 0.005–0.32; KSD 0.05–3.2;
  sliced KSD 0.625–40; MMD 2.5–160; VICReg 0.3125–20; covk = cov. Linear rung (`LAM2_LIN`): the same
  plus one level above (cov to 2.56, KSD to 6.4). Secondary E1 settings use level indices
  α = 0.5: 0–4, α = 0.25: 0–4, d = 6: 3–6, d = 24: 1–4. E3: KSD {0.8, 3.2}, SIGReg {0.08, 0.32} at
  8000 steps. E4: KSD λ {0.1, 0.2}, VICReg {1.25, 5} × σ {0.5, 1} × {batch, fixed}. E5: study 1's H3 cells
  (uniform: KSD 0.2, matched-uniform KSD 0.2, SIGReg 0.05, MMD 10; Laplace: KSD 0.2, matched-Laplace KSD 1.0,
  SIGReg 0.01, MMD 10), each also at λ/5. E6: covk on the cov grid.
- **P2 run-count estimate** (`s2_all`: 1110 runs, 5 held-out seeds; hours = per-arm pilot times / 6
  workers), blocks in priority order:
  | block | runs | est. h | cumulative h |
  |---|---|---|---|
  | E1_core (6 arms × 7 λ, α 1, d 12) | 210 | 1.22 | 1.22 |
  | E3_long (8000 steps) | 20 | 0.48 | 1.70 |
  | E2_linear (cov, KSD × 8 λ × α {1, 0.5}) | 160 | 0.64 | 2.34 |
  | E1_alpha0.5 | 150 | 0.87 | 3.21 |
  | E1_d6 | 120 | 0.70 | 3.90 |
  | E1_alpha0.25 | 150 | 0.87 | 4.77 |
  | E1_d24 | 120 | 0.70 | 5.47 |
  | E4_noise | 80 | 0.43 | 5.90 |
  | E5_ident | 80 | 0.49 | 6.38 |
  | E6_covk | 35 | 0.12 | 6.50 |
  With a 6 h allowance and a 30 % margin (4.6 h), everything through E1_alpha0.25 fits in the first
  handoff (≈ 4.8 h estimated); the rest (d = 24, E4, E5, E6; ≈ 1.7 h) runs in the second handoff if
  allowance remains. A cut leaves the lowest-priority blocks, in the prompt's cut order.

## Decisions
- D1. `eval_v=2` for every study-2 run: computes the new metrics, skips study 1's NP-KSD (unused, slow).
  It changes the hash, so study-1 cached runs are re-trained rather than reused (they lack the variance
  shares anyway); `eval_v` does not change training (validation B: identical metrics).
- D2. `linear` predictor initialised at I + 0.1·N(0, 1/d), bias 0 (as in `theory/check_t1.py`).
- D3. `covk` (E6) = Σ_{i≤k}(ev_i − 1)² over the k = 6 largest batch-covariance eigenvalues (float64 on CPU).
- D4. Variance shares are held out (OLS fit on the 10 000 probe-fit images, evaluated on the 5 000 eval
  images), pooled over coordinates in raw embedding units; `_oh` versions regress on one-hot factor levels
  (additive per factor). `nuis_dir_ratio` = mean of the top min(5, d−6) `nuis_eigs` / mean of the top
  min(6, d) `content_eigs` (the theory's v; undefined for d ≤ 6). P9 reads this ratio.
- D5. Theory-meets-code gate statistic = check_t1's minibatch statistic (mean v and c, tol 0.03), with
  empty-direction traps counted and reported (GATE_0 D5). I first required every *single* active
  eigenvalue within 0.03; minibatch scatter of single eigenvalues (up to ~0.05, also in the content
  eigenvalues) fails that at q = 5, so the gate uses the mean over active directions. Both versions are
  visible in `stein/validate2.py`'s docstring and here.
- D6. **Supplementary threshold estimator** (defined now, before any held-out study-2 run): the frozen
  estimator (`theory/predictions.md`) is reported as primary. Because the pilot shows leakage that is
  high at the lowest λ for some arms (cov 0.01–0.02, sliced KSD 0.5, VICReg ≤ 0.5, linear cov 0.01: a
  low-λ regime with low content R²), `analyze2.py` also reports the same crossing searched upward from the
  λ of minimum mean leakage (`upward=True`). The grids keep their low levels (they show the regime).
- D7. E3's λ levels: about 2–3× and 8–10× the pilot onset; E4's KSD λ = 0.1 (≈ λ̂*(0)/2 from the pilot)
  and 0.2 (study 1's level, P14); VICReg 1.25 and 5 (study 1's level). All E3/E4 σ = 0 or 2000-step
  partners are on the E1 grid, so they are shared by hash.
- D8. Synthetic runs use `device=cpu`; `theory/_lib.py` sets float64 as the global default, so
  `validate2.py` resets float32 after importing it (the first validation attempt ran synth in float64;
  those 29 runs were deleted from `runs/` and rerun).
- D9. No new packages (`requirements.txt` unchanged; pandas' `to_markdown` avoided since `tabulate` is
  not installed).

## Problems / findings for P2–P3
- **d = k = 6 in the code gate (observation; originally written as a gate, failed):** Theorem 1 cor. c
  says no *global* minimiser holds nuisance at d ≤ k. The real loop (synth, linear encoder, linear
  predictor, 4000 steps) ends with nuisance in one direction and one content direction dropped
  (content R² ≈ 5/6) in 0/6 seeds at q = 2, 5/6 at q = 5 and 6/6 at q = 20; at q = 5 it persisted to
  32 000 steps, at q = 20 it was gone by 32 000 steps (seed 0). So the training dynamics can trade a
  content direction for nuisance at d = k; the global statement is not contradicted, but P5 (no
  leakage at d = 6 in networks) may fail for this reason. Mechanism not established (a quick
  second-order calculation suggests a near-degenerate stationary point; not checked — heuristic only).
- Near threshold (q = 1.25), 1 of 6 seeds (seeds 0–5) ends with one of five nuisance directions empty
  (`near_threshold_extra_seeds` in the json): Prop. 1's traps also appear in the real loop.
- Observations: the MLP predictor reproduces the linear-predictor values at α = 1 (v = 0.175, 0.468, 0.761
  at q = 1.25, 2, 5); at α = 0.5 the real loop follows the *global* (learned-predictor) solution, with
  onset below λ* = c_p/2 (v = 0.073/0.091 vs reference 0.093 at q = 0.85), not the trap threshold.
- In the synth tier the probe R² is ≈ 0.87 even below threshold (nuisance eigenvalues ~0.001 that Adam
  leaves, decoded perfectly in noise-free data: cor. a's scale invariance). Use the variance share and
  eigenvalues there, not probe R².
- Linear-encoder onsets in the pilot lie above the conv ones (cov > 0.16, KSD 1–5), outside P8's
  factor-3 window for cov if confirmed — to be read on held-out seeds in P2.

## Next (P2)
- Review and test `study2/jobs/P2.sh` (smoke test already run: `s2_tiny`, 10 configs × 30 steps, all
  blocks' configs train and evaluate, analysis runs), hand it off, then analyse with
  `stein/analyze2.py`, write verdicts per prediction.
