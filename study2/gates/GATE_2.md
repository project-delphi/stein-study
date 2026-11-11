# GATE 2 — Experiments and analysis (study 2) — FINAL

## Done
- **Runs.** 1075 of the 1110 planned held-out runs (seeds 1–5) finished with 0 failures. Every block is
  complete except E6 (35 runs), which was cut (see D2). Handoff 1 (`study2/jobs/P2.sh`) ran for the full
  6 h allowance and finished E1_core, E3, E2_linear, E1 α 0.5 / d 6 / α 0.25 and 107/120 of E1 d 24. Two
  foreground commands, each under the 45-min limit, then ran the rest of d 24 plus E4 (200 configs, 13 of
  them new d-24 runs plus 80 E4 runs; log `study2/logs/p2_foreground.log`) and E5 (65 new runs,
  `study2/logs/p2_foreground2.log`).
- **Analysis.** `.venv/bin/python -W ignore -m stein.analyze2` writes `results2/runs2.parquet`, `cells2.parquet`,
  `thresholds.parquet`, `frontier.parquet`, `schema.json` and `analysis.md`.
  `.venv/bin/python -W ignore -m stein.verdicts2` (new) writes `results2/verdicts.json`, which holds the
  statistic behind each prediction. **Every number below comes from these files.** Both are
  deterministic (seeded bootstrap, 2000 draws, seeds resampled). 40 tests pass.

## Estimators (fixed before any held-out study-2 run was seen)
- Primary threshold: as in `theory/predictions.md` (the smallest λ at which the 5-seed mean linear
  leakage R² exceeds 0.10, interpolated in log λ, 2000-draw seed bootstrap with the same draw at every λ,
  censored as "<"/">"). Secondary: nuisance variance share with τ = 0.05. Supplementary (GATE_1 D6): the
  primary search started upward from the λ of minimum mean leakage.
- Frontier: the best 5-seed mean content R² along the path subject to mean leakage ≤ 0.05/0.10/0.20, with
  a bootstrap CI. Also the leakage at matched content R² (0.5, 0.4), interpolated along the path after
  the arm's content maximum (`verdicts2.py`).

## Key tables (conv encoder, uniform law, 2000 steps unless stated)
λ̂* (primary, 95 % CI) at α = 1, d = 12, against the linear model (GATE_0):
| arm | λ̂* | CI | theory | ratio |
|---|---|---|---|---|
| cov | < 0.02 (censored) | – | 0.043 | – |
| SIGReg | 0.0282 | [0.0262, 0.0321] | 0.039 | 0.72 |
| KSD | 0.296 | [0.272, 0.345] | 0.94 | 0.32 |
| sliced KSD | 2.15 | [1.62, 2.78] | 7.3 | 0.29 |
| MMD | 13.6 | [12.5, 16.0] | 17.1 | 0.79 |
| VICReg | < 0.3125 (censored) | – | ~7 (heuristic) | – |

Censored cov and VICReg: at the lowest λ both are in a collapse regime. Cov at λ = 0.02 has embedding
std ≈ 0.000, content R² 0.05 and mean leakage 0.102. VICReg at 0.3125 has effective rank 2.4, content
R² 0.24 and leakage 0.101. A scale-invariant probe still reads nuisance from such an embedding (cor. a),
so their leakage paths start above τ, and neither the primary nor the upward variant can place an onset.

Ratio λ̂*(α)/λ̂*(1), with CIs:
| arm | α 0.5 | α 0.25 |
|---|---|---|
| SIGReg | 0.92 [0.80, 0.99] | 0.52 [0.43, 0.70] |
| KSD | 1.27 [0.94, 1.57] | 0.55 [0.44, 0.75] |
| sliced KSD | 1.20 [0.77, 1.65] | 0.65 [0.50, 0.86] |
| MMD | 0.94 [0.79, 1.10] | 0.50 [0.41, 0.63] |

Frontier at α = 1, d = 12: best content R² with leakage ≤ 0.05 (the same at ≤ 0.10):
KSD 0.727 [0.715, 0.748] > MMD 0.629 [0.603, 0.661] > SIGReg 0.573 [0.528, 0.618] ≈ sliced KSD 0.572
[0.534, 0.637] > cov (none ≤ 0.10; 0.385 at ≤ 0.20) > VICReg (none ≤ 0.05; 0.258 at ≤ 0.20).
Leakage at matched content R² 0.5: MMD 0.16 [0.14, 0.19], SIGReg 0.21 [0.10, 0.27], KSD 0.45 [0.40, 0.49],
sliced KSD 0.45 [0.28, 0.51]. Cov and VICReg never reach 0.5 (their content maxima are 0.45 and 0.40).

## Verdicts on the pre-registered predictions
| # | verdict | evidence (`results2/verdicts.json` key) |
|---|---|---|
| P1 step-like onset | **refuted** (registered criterion) | `P1`: cov and VICReg have leakage ≥ 0.10 at the lowest λ (0.102, 0.101; the collapse regime above) although their theory thresholds lie inside the grid. The four kernel arms do show a step: < 0.10 to ≥ 50 % of the maximum within 0.30–0.60 decades (SIGReg 0.30, KSD 0.60, sliced KSD 0.60, MMD 0.30). MMD's lowest-λ mean (−7.7) is a probe blow-up on collapsed seeds. |
| P2 location within ×3 | **refuted** | `P2`: KSD 0.32× and sliced KSD 0.29× theory (just outside ×3); SIGReg 0.72× and MMD 0.79× are inside; cov and VICReg are censored. Every uncensored threshold lies *below* the linear model. |
| P3 ordering | **confirmed** for SIGReg < KSD < sliced KSD < MMD; cov ≈ SIGReg **inconclusive** (cov censored) | `P3`: KSD/SIGReg 10.5 [8.9, 12.5], sliced KSD/KSD 7.3 [5.3, 9.6], MMD/sliced KSD 6.3 [4.8, 8.8]; all CIs exclude 1. |
| P4 α scaling in [α(2−α), 1], not ∝ α | **confirmed** for SIGReg, KSD, sliced KSD, MMD; cov and VICReg censored | `P4`: ratios at α = 0.25 are 0.50–0.65, all CIs ≥ 0.41 (∝ α would give 0.25), close to the learned-predictor value 0.44. At α = 0.5 two arms sit above 1 (KSD 1.27, sliced KSD 1.20), but their CIs include values in [0.75, 1]. Not monotone in α for those two. |
| P5 no threshold at d = 6 | **refuted** | `P5`: all 6 arms reach leakage ≥ 0.10 at d = 6 (maxima 0.33–0.67). Leakage at d = 6 is below its d = 12 value at every shared λ for every arm, and onsets move up (KSD 1.24 vs 0.30; sliced KSD 5.7 vs 2.15; cov 0.36). Content R² falls as leakage rises, the content-for-nuisance trade GATE_1 saw at d = k in the synthetic loop. |
| P6 d = 24 ratios | **not refuted by its registered criterion; point values miss** | `P6`: KSD 0.36 [0.31, 0.40] (predicted ≈ 0.63); SIGReg 0.76 and MMD 0.81 [0.64, 0.90] (predicted 0.9–1.2; the criterion required < 0.6 to count against); sliced KSD < 0.58 (censored); cov censored. Wider embeddings lower every threshold, by more than predicted. |
| P7 probe steeper than share | **confirmed (weak)** | `P7`: the 10–90 % rise is narrower for leakage R² in 4 of 6 arms. MMD 0.25 vs 0.81 decades is clear; KSD 0.70/0.74, SIGReg 0.60/0.62 and sliced KSD 0.62/0.74 are small differences. Cov and VICReg have no clean onset. The share crosses 0.05 at a λ 2–3× above the probe's 0.10 crossing (KSD 0.81 vs 0.30, MMD 32.6 vs 13.6): small nuisance variance is already decodable. |
| P8 linear rung | **refuted** | `P8`: cov λ̂* = 0.40 [0.37, 0.46], 9.4× theory; KSD 1.25 [1.11, 1.60], 1.33×, which is inside. α ratio: cov 0.69 [0.55, 0.87], KSD 1.06 [0.82, 1.35]. |
| P9 functional form | cov **inconclusive**; kernel part **confirmed** | `P9`: conv cov's primary λ̂* is censored. With the secondary (share) estimate 0.093, the per-direction ratio v is 0.63 at 2λ̂* (in range) and 1.44 at 5λ̂*: nuisance directions end up *larger* than content directions (v > 1, impossible in the linear model). Linear-rung cov: v = 0.30 at 2λ̂* (lower edge of the range) and 0.53 at 5λ̂* (below [0.6, 1]). Both points are below the theory, but the registered "outside both ranges" is not met. KSD/SIGReg/MMD at 2λ̂*: 0.11/0.21/0.18, slower than 0.5 as predicted. |
| P10 KSD/cov > 5 | **refuted** on the linear rung; conv inconclusive | `P10`: linear rung 3.1 [2.6, 4.0]. Conv: cov censored on the primary; the secondary share estimator gives 8.7 [7.1, 38.8]. |
| P11 leakage persists to 8000 steps | **refuted** | `E3`: leakage falls at all four cells. 8000 − 2000 steps: KSD 0.8 −0.36 [−0.40, −0.32], KSD 3.2 −0.41 [−0.50, −0.30], SIGReg 0.08 −0.33 [−0.47, −0.24], SIGReg 0.32 −0.18 [−0.32, −0.03]. At KSD 0.8 and SIGReg 0.08 it falls to ≈ 0.00. |
| P12 what fills the spare directions (descriptive) | – | `E3`: with longer training the linearly unexplained share rises (KSD 0.8: 0.59 → 0.84; one-hot-unexplained 0.34 → 0.58). Linear content R² falls (0.60 → 0.32) while kNN content R² rises to 1.00 and kNN leakage falls (0.64 → 0.19). The network replaces nuisance with *nonlinear* functions of content, the zero-cost, nuisance-free solution of Theorem 1 cor. d, which lies outside the linear model. |
| P13 sign of σ (KSD) | **confirmed** for leakage; content-share sub-claim **not borne out** | `E4`: batch σ raises KSD leakage at both λ (0.1: +0.023 [0.016, 0.031] at σ 0.5, +0.66 at σ 1; 0.2: +0.30 / +0.65), all CIs > 0. The content share does not fall (λ 0.2: 0.44 → 0.52 → 0.51). |
| P14 magnitude (KSD) | **refuted in magnitude** at λ = 0.2 (as registered); **pattern confirmed** at λ = 0.1 | `E4`: at λ = 0.2 the linear model predicts no leakage at σ = 1; observed +0.646 [0.596, 0.687], identical to study 1's H2 contrast, as expected since the config is the same and training is deterministic. At λ = 0.1 (≈ λ̂*(0)/3): onset at σ = 1 (0.66), none at σ = 0.5 (0.024 < τ). |
| P15 fixed vs batch | **refuted** (KSD) | `E4`: fixed σ = 0.5 raises KSD leakage *more* than batch σ = 0.5 (fixed − batch: +0.13 [0.04, 0.24] at λ 0.1, +0.15 [0.04, 0.26] at 0.2). Fixed ≈ batch at σ = 1 (CI includes 0). KSD *shrinks* the embedding under fixed σ = 1 (std 0.80 → 0.41). VICReg: no inflation (fixed − batch emb. std: +0.03 [−0.00, 0.06], +0.01 [−0.04, 0.05]) and no σ effect on leakage (all CIs include 0, point values negative). VICReg part inconclusive. |
| P16 orth-fit R² no matched-prior gain | **confirmed** | `E5`: matched − Gaussian KSD, uniform: −0.022 [−0.083, 0.036] at λ 0.2, +0.026 [−0.053, 0.109] at 0.04. Laplace (matched 1.0 vs Gaussian 0.2; and 0.2 vs 0.04): −0.50 and −0.44, but there the matched-Laplace runs lose content (content R² −0.49) and leak (+0.33). |
| P17 no hue-pair MCC gain | **refuted** | `E5`: uniform λ 0.2: matched-prior MCC +0.087 [0.015, 0.145] overall, hue pair +0.25 [0.18, 0.32], scale +0.22 [0.11, 0.33]. At λ 0.04 there is no clear gain (hue +0.08 [−0.01, 0.18]). Laplace hue +0.12 [0.04, 0.22] at the selected levels. So MCC sees a matched-prior gain that orth-fit R² misses (T4(i)'s point about the metric holds), but the gain is not confined to the independent scalar factor, and at uniform λ 0.2 it comes with more leakage (+0.12) and lower content R² (−0.06). |
| P18 covk | **not tested** (E6 cut) | – |

Not predicted (reported descriptively): the frontier ranking (above); kNN leakage (0.13–0.22 below the
kernel arms' onsets); content R² levels.

## Answers
- **Study 1's finding 1 (leakage follows λ): explained in form, not in detail.** Every arm with a
  measurable onset has a threshold-like rise within ≤ 0.6 decades. The thresholds order across
  regularisers as T2's slope at a collapsed direction predicts (SIGReg < KSD < sliced KSD < MMD, each step
  by a factor 6–10). They scale with α as α(2−α) rather than α, and fall with d. Locations are off by
  0.3–0.8× (always earlier than the linear model), and with a linear pixel encoder by up to 9× the other
  way (cov). The step is not forced. E3 shows it is a transient of 2000-step training: at 8000 steps the
  network fills the spare directions with nonlinear content features and leakage falls to ≈ 0. Cor. d
  allows exactly this, and the linear model cannot produce it. So the theory explains *where* the
  2000-step network leaks, and its own cor. d explains why that is not an equilibrium.
- **Finding 2 (σ raised leakage): sign explained for KSD, magnitude not.** Batch-scaled σ lowers KSD's
  threshold, as T3 says, but much more strongly than the linear model predicts (leakage 0.65 at λ = 0.2,
  where it predicts none). The predicted ordering of fixed-scale against batch-scaled noise is wrong at
  σ = 0.5. For VICReg, σ does not raise leakage at all in this grid. That narrows rather than explains
  study 1's "σ never helps".
- **Finding 3 (no gain from a matched prior): explained as a metric artefact, partly.** Orth-fit R² shows
  no gain (P16 confirmed, as T4(i) requires). MCC does show a gain for the matched uniform prior at
  study 1's λ (+0.09 overall, +0.25 on the hue pair). T4(ii) did not predict the hue-pair part (hue's
  cos/sin are dependent), and the gain comes with more leakage.
- **Ranking by frontier.** Yes, the ranking changes. Study 1's H1 had tuned SIGReg leaking more than
  VICReg at VICReg's least-leaky λ. On the frontier VICReg is last: it never has leakage ≤ 0.05 with any
  content, and its whole path stays at content R² ≤ 0.40. KSD reaches the most nuisance-free content
  (0.73 with leakage ≤ 0.05), then MMD (0.63), then SIGReg ≈ sliced KSD (0.57). Above the thresholds the
  order at matched content differs again (MMD and SIGReg leak least at content 0.5), so "which arm leaks
  least" depends on the operating point, not on the target law.

## Decisions
- D1. Threshold and frontier paths use only the registered λ grids (cells in blocks E1_*, E2_*, E6_*).
  `stein/analyze2.py` was edited so that E5 cells at the base setting with off-grid λ (KSD 0.04, SIGReg
  0.05, MMD 2) do not add levels to the E1 paths. The edit was made before any held-out result existed.
- D2. The handoff allowance ran out with 196 runs left. Following the prompt's cut order, I ran E1 d 24
  (13 runs), E4 (80) and E5 (65) as two foreground commands, each within the 45-minute command limit
  (about 37 and 25 min), because they carry registered predictions P6 and P13–P17. **E6 (P18), the
  first item in the cut order, was cut.** The second handoff was not used: it had no allowance left.
- D3. Censored thresholds (cov, VICReg at α = 1, d = 12) are reported as censored. Where a prediction
  depends on them (P3 cov ≈ SIGReg, P9 cov, P10 conv), the verdict is "inconclusive". The secondary
  (share) estimator is shown alongside, labelled as such, and does not decide a verdict.
- D4. The E5 Laplace comparison pairs study 1's selected levels (Gaussian KSD 0.2 against matched 1.0,
  and both at λ/5), as registered. Uniform pairs equal λ.
- D5. P6 is judged by its registered "against" criterion (not met) and the miss of the point values is
  reported beside it, rather than calling it confirmed.

## Problems / surprises (for P3)
- The biggest surprise is E3: leakage is a 2000-step transient. The paper should present the threshold
  as a property of the regime that training reaches, and cor. d as the explanation for the decay.
- The low-λ collapse regime of cov and VICReg (probe R² ≈ 0.10 on a near-zero or rank-2 embedding)
  makes their thresholds unmeasurable with the frozen estimator.
- Conv cov's per-direction nuisance ratio exceeds 1 at high λ (1.8 at λ 0.64, 2.2 at 1.28): nuisance
  directions carry more variance than content directions. That is outside the linear model.
- The linear pixel encoder's cov threshold is 9× the theory: content is not a linear function of
  pixels, so content is not free for a pixel-linear map. That breaks the premise of T1. Not analysed further.
- kNN leakage is 0.13–0.22 below the onset for the four kernel arms (except MMD at λ 2.5, 0.05), so
  nonlinear nuisance information is present even where linear leakage is ≈ 0. The theory covers linear leakage only.

## Next (P3)
- Paper figures: λ paths with the T1/T2 curves and thresholds (E1), the α-ratio plot against
  α(2−α) (P4), frontier curves, E3 2000 vs 8000 steps with the variance decomposition, and E4/E5
  contrasts. Read the numbers from `results2/verdicts.json`, `thresholds.parquet`, `frontier.parquet` and
  `cells2.parquet`.
