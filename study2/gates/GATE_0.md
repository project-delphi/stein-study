# GATE 0 — Theory, predictions, literature (study 2) — FINAL

## Done
- `theory/theory.md`: statements, proofs and status of T1–T4 (+ Lemma 1, Theorem 2, Props 1 and 3).
- `theory/_lib.py`: population (expected batch) versions of cov, KSD-IMQ, sliced KSD, SIGReg (Epps–Pulley
  incl. its 1/N bias, N = 256), MMD and VICReg, validated against `stein/losses.py` by Monte Carlo.
- Checks (each writes `results2/theory_<name>.json`; run `.venv/bin/python theory/check_t<i>.py`):
  | check | runtime | result |
  |---|---|---|
  | check_t1.py | ~4 min | PASS (A identity, α=0, B fast learned predictor, C minibatch code units, D nonlinear) |
  | check_t2.py | ~5 min | PASS (MC validation; GD vs eigenvalue solution for 5 arms × α ∈ {0.5,1} × 3 λ); VICReg scan (heuristic) |
  | check_t3.py | ~17 min | json `pass: false` by design of its tolerance: fixed-scale noise matches (cov, KSD), batch-scaled matches for cov but not KSD (A-diag fails); T3 is labelled heuristic accordingly |
  | check_t4.py | ~5 min | PASS (invariance; Comon global claim; orth-fit R² ≈ 1) |
- `theory/predictions.md`: P1–P18 for E1–E6, the λ̂* estimator (τ = 0.10 linear leakage, log-interp,
  seed bootstrap), and a list of what the theory does not predict. **Frozen from here on.**
- Literature (8 lit-reviewer notes, all VERIFIED): tian2021understanding, jing2022understanding,
  simon2023stepwise, balestriero2022contrastive, shwartzziv2023information, comon1994independent,
  zimmermann2021contrastive, shah2020pitfalls; `notes/related_theory.md` written.

## Status of each statement
- T1: **corrected and proved.** λ* = c_p·α (identity predictor, as sketched) but λ* = c_p·α(2−α)/2 with a
  learned linear predictor; m < d−k leaves d−k−m directions collapsed; corollaries a–d proved (d: the
  threshold needs linearity). Prop. 1 (empty-direction traps; escape at c_p/2 independent of α): heuristic,
  numerically supported (21/60 joint-W and 6/135 fast-predictor GD runs trapped, all satisfying c(u) ≥ 2λ).
- T2: scaling λ*_reg = c_p α(2−α)/t*_reg **proved** for any Σ-dependent regulariser; t*_reg **numerical**:
  λ* (code units, d=12, α=1): cov 0.0427, SIGReg 0.0389, KSD 0.939, sliced KSD 7.33, MMD 17.1, VICReg in
  (4.6, 10] (heuristic). The second-order "curvature at I" proxy is **refuted** as a threshold predictor
  (off by up to 5×, 12× at d = 24); the slope at a collapsed direction sets it.
- T3: **heuristic.** cov: σ leaves λ* unchanged; kernel arms: batch-scaled σ lowers λ* (KSD 0.94→0.25 at
  σ=1), which gives study 1's sign but not its size at λ = 0.2.
- T4: (i) invariance **proved**; (ii) Comon 1994 Thm 11 applied (**proved by citation**, idealised:
  "vanishes only at the prior" assumed; discrete content). Scale factor is the cell where T4 fits best.
- Novelty: T4 is known (Comon; Zimmermann et al. closest analogue). T1–T3 not found in the 8 papers; to be
  claimed as derivations/tests in this setting.

## Decisions
- D1. Units: `F.mse_loss` averages over d, so c_p = 1/d in code units; theory stated with general c_p.
- D2. Study 1's non-EMA arms use no stop-gradient and a learned MLP predictor → analysed the jointly
  trained (and the fast, profiled-out) linear predictor; stop-gradient variant not analysed.
- D3. Real nuisance resampling is per factor (3 factors, 5-dim code); the (1−α) structure holds per factor
  for any single-factor feature (Lemma 1 remark).
- D4. cov arm for P1: R_cov(z) = ‖Σ̂ − I‖²_F, Σ̂ the centred unbiased batch covariance, per view, averaged
  over the two views (expected value includes (‖Σ‖²_F + (trΣ)²)/(N−1)).
- D5. GD checks of global claims use best-of-starts plus an explicit closed-form minimiser, and require
  every non-global start to be a verified trap; traps are reported as a finding, not hidden.
- D6. T3 batch-scaled noise analysed under A-diag (principal axes = coordinates); labelled heuristic since
  minibatch KSD training does not satisfy it.
- D7. Zimmermann BibTeX year set to 2021 (arXiv 2102.*, ICML 2021) with a note; the arXiv record says 2022 (v4).
- D8. No new packages installed (scipy already present); `requirements.txt` unchanged.

## Problems
- jing2022understanding note: section pointers were from memory (flagged in the note); cite main claim only.
- simon2023stepwise, shwartzziv2023information read from the abstract only; balestriero2022 abstract +
  keyword scan; shah2020pitfalls BibTeX lacks volume/pages (not verified).
- The matched uniform-surrogate KSD prior has spurious rotated local minima in the linear ICA check
  (3/3 seeds), relevant to E5's interpretation.

## Next (P1)
- Implement arm `cov` exactly as D4, encoder `linear`, metrics (variance shares, MCC with per-factor
  values incl. scale, per-principal-direction nuisance variance), noise mode `fixed`.
- Theory-meets-code gate: use α = 1 (global and trap thresholds coincide): cov + linear encoder on synthetic
  linear-Gaussian views, k = 6, m = 5, d = 12, batch 256 → onset at λ* = 1/(2d) = 0.0417 (0.0427 with the
  finite-N term); per-direction nuisance variance ≈ 1 − λ*/λ. At α < 1 expect onset between
  c_pα(2−α)/2 and c_p/2 (MLP predictor, Prop. 1).
- P2 λ grids should bracket the T2 values: cov ~0.01–0.3, SIGReg ~0.005–0.3, KSD ~0.1–5, sliced KSD
  ~0.5–50, MMD ~2–100, VICReg ~0.2–50 (VICReg's network onset was already < 1 in study 1).
- The threshold estimator is fixed in `theory/predictions.md`; do not redefine it.
