# Pre-registered predictions for study 2 (written in P0, before any study-2 network was trained)

**This file is frozen after P0.** Later phases report every prediction's outcome and do not edit it.
Theory references are to `theory/theory.md`; linear-model numbers come from `results2/theory_t*.json`.
Code units throughout (prediction loss = per-coordinate MSE, so c_p = 1/d). k = 6 content dims and m = 5
nuisance dims in the probe codes. Leakage = linear-probe R² of the nuisance code unless stated otherwise.

## Threshold estimator (fixed now)
λ̂* = the smallest λ at which the 5-seed mean linear leakage R² exceeds **τ = 0.10**, interpolated
linearly in log λ between adjacent grid levels. 95 % CI: percentile bootstrap over seeds (resample the 5
seeds with replacement, recompute λ̂*, 2000 draws). Reported as "< λ_min" or "> λ_max" when the crossing
lies outside the grid. Secondary: the same with the nuisance variance share and τ = 0.05.
"Within a factor F" means F⁻¹ ≤ λ̂*/λ_theory ≤ F.

## E1 — λ paths (arms cov, VICReg, SIGReg, sliced KSD, KSD, MMD)
- **P1 (step-like onset).** For every arm at α = 1, d = 12, leakage is < 0.05 at the lowest grid λ and
  goes from < 0.10 to ≥ 50 % of its maximum over the grid within one decade of λ. *Against:* a rise spread
  over more than one decade, or leakage ≥ 0.10 at the lowest level for an arm whose theory threshold
  lies inside the grid.
- **P2 (location).** λ̂*(α = 1, d = 12) is within a factor 3 of the linear-model value: cov 0.043,
  SIGReg 0.039, KSD 0.94, sliced KSD 7.3, MMD 17. VICReg (heuristic value in (4.6, 10]) is within a
  factor 3 of 7 — low confidence, since study 1 already saw leakage 0.17 at λ = 1. *Against:* outside
  the factor for ≥ 2 of the 5 Gaussian-target/cov arms.
- **P3 (ordering).** λ̂*: SIGReg ≈ cov (ratio within a factor 2) < KSD < sliced KSD < MMD.
  *Against:* any inversion in that chain whose bootstrap CIs do not overlap.
- **P4 (α scaling — the sketch's λ* ∝ α is predicted to FAIL).** λ̂*(α)/λ̂*(1) lies in [α(2−α), 1]:
  [0.44, 1] at α = 0.25 and [0.75, 1] at α = 0.5 (lower end: global optimum with a learned predictor;
  upper end: empty-direction traps, Prop. 1). *Against:* the ratio at α = 0.25 below 0.35 (closer to
  α = 0.25, supporting ∝ α) or above 1.2, for a majority of arms.
- **P5 (d ≤ k).** At d = 6 (= k) no arm shows a threshold: leakage stays < 0.10 at every λ in the grid
  and below its d = 12 value at every λ. *Against:* leakage ≥ 0.10 at any λ for any arm (report how many).
- **P6 (d = 24).** λ̂*(d = 24)/λ̂*(d = 12) ≈ 0.5 for cov, ≈ 0.63 for KSD, and 0.9–1.2 for SIGReg,
  sliced KSD and MMD. *Against:* cov ratio > 0.8, or SIGReg/MMD ratio < 0.6.
- **P7 (two leakage metrics).** Over the λ path the linear leakage R² rises more steeply (narrower
  log-λ interval from 10 % to 90 % of its range) than the nuisance variance share, which starts at the
  same λ (Theorem 1 cor. a). *Against:* the share rising as steeply or more steeply for most arms.
- **Not predicted:** the content–leakage frontier ranking of arms; content R² levels; kNN leakage.
  (The theory places thresholds, not frontiers.)

## E2 — quantitative tests (incl. the pixel-linear encoder, arms cov and KSD, α ∈ {0.5, 1})
- **P8 (linear rung).** With the linear encoder, λ̂* for cov is within a factor 3 of 0.043 (α = 1) and
  for KSD within a factor 3 of 0.94; the α = 0.5 / α = 1 ratio is in [0.75, 1] for both.
  *Against:* either location or the ratio outside these ranges.
- **P9 (functional form above threshold).** For cov (conv encoder and linear rung), the mean nuisance
  variance per nuisance-dominated principal direction at λ ≈ 2λ̂* is in [0.3, 0.7] of the content
  per-direction variance (theory 0.5; v = 1 − λ*/λ), and at λ ≈ 5λ̂* in [0.6, 1.0] (theory 0.8). For
  KSD/SIGReg/MMD the rise near λ̂* is slower than 1 − λ̂*/λ (theory at 2λ*: 0.22 / 0.40 / 0.37).
  *Against:* cov values outside both ranges.
- **P10 (orders of c_reg).** Same as P3, read on the E2 slice (cov vs KSD): λ̂*_KSD/λ̂*_cov ≈ 22
  (linear model), predicted > 5. *Against:* ratio < 5.

## E3 — training length (KSD, SIGReg; two λ above threshold; 2000 vs 8000 steps)
- The theory does **not** say which zero-cost solution a nonlinear network finds (Theorem 1 cor. d).
- **P11 (weak, from the trap mechanism).** Leakage persists: leakage at 8000 steps ≥ leakage at 2000
  steps − 0.05 at both λ levels for both arms. *Against:* a drop > 0.05 whose CI excludes 0.
- **P12 (what fills the spare directions).** No numeric prediction; the linear model says spare
  directions hold content (k), nuisance (r) and nothing else, so a large unexplained share in the
  network would be outside the linear theory. Reported descriptively.

## E4 — noise bottleneck (KSD, VICReg; d = 12; batch-scaled vs fixed-scale σ ∈ {0, 0.5, 1})
- **P13 (sign, KSD).** At every λ tested, KSD leakage is non-decreasing in batch-scaled σ, and the
  content share of variance falls with σ. *Against:* leakage falling with σ (CI excluding 0) at any λ.
- **P14 (magnitude, KSD).** λ*(σ)/λ*(0) for batch-scaled σ = 0.5 / 1 is 0.84 / 0.26 (linear model). At
  λ = 0.2 (study 1's tuned level) the linear model predicts **no** leakage even at σ = 1 (λ*(1) = 0.25):
  this part is already contradicted by study 1 (H2: +0.646), and is registered so that the paper reports
  it as a failure of the magnitude. At λ = λ̂*(0)/2 it predicts onset at σ = 1 but not at σ = 0.5.
- **P15 (fixed vs batch).** For KSD, fixed-scale σ = 0.5 raises leakage less than batch-scaled σ = 0.5
  (threshold shifts 6 % vs 16 %); fixed σ = 1 raises it at least as much as batch σ = 1. For VICReg,
  fixed-scale noise can be escaped by inflating the embedding (the hinge has no upper limit): embedding
  std rises with fixed σ but not with batch σ, and fixed σ raises VICReg leakage less than batch σ
  (heuristic). *Against:* the reverse orderings with non-overlapping CIs.

## E5 — identifiability with MCC (study 1's H3 cells; Laplace and uniform laws)
- **P16.** R² after an orthogonal fit differs by < 0.02, or favours the Gaussian target, between
  matched-prior KSD and Gaussian-target KSD at each law and λ (Prop. 3 and its orbit argument).
  *Against:* matched prior better by ≥ 0.02 with CI excluding 0.
- **P17.** A matched-prior MCC gain, if present, is concentrated in the scale factor (the only
  independent scalar factor): matched-prior KSD's per-factor MCC for the hue pair is not higher than
  Gaussian-target KSD's (CI includes 0 or favours Gaussian). Gaussian-target arms (KSD, SIGReg, MMD)
  have MCC below 1 with no systematic ordering implied by the theory. The overall MCC gain may be null,
  because training need not find the identifying solution (T4 check: uniform-surrogate KSD stalled at a
  rotated solution in 3/3 seeds). *Against:* a hue-pair MCC gain for the matched prior with CI excluding 0.

## E6 — optional theory-guided arm (first to be cut)
- **P18.** A covariance penalty applied only to the top-k principal subspace (spare directions left
  unconstrained) has no threshold in the linear model (nuisance has a cost and no benefit), so in the
  network its leakage stays below the cov arm's at every λ where the cov arm is above threshold, at a
  content R² no lower than cov's at the same λ. *Against:* leakage at or above cov's at ≥ 2 such λ.

## What the theory does not predict
Which zero-cost solution a nonlinear network finds; the size of leakage above threshold in a network
(the linear model's probe R² jumps to r/m = 1); content R² values; kNN-probe leakage; anything about EMA
or the temporal tier; the magnitude of study 1's σ effect at λ = 0.2 (see P14); whether a matched prior is
found by training.
