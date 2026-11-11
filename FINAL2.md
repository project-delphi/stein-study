# FINAL2 — Study 2: a linear theory of JEPA regularisers and its test

Paper: `paper2/paper.pdf` (from `paper2/paper.qmd`). Reproduction: `REPRODUCE2.md`. Phase records:
`study2/gates/GATE_0.md` – `GATE_4.md`. Every number below comes from `results2/` (statistics in
`results2/verdicts.json`, `thresholds.parquet`, `frontier.parquet`, `cells2.parquet`; theory checks in
`results2/theory_t*.json`) through the scripts named in the gate files.

## Summary
Study 1 left three findings unexplained: leakage follows λ, predictor-input noise raised leakage, and a
matched prior did not improve orthogonal-fit identifiability. Study 2 derived a linear content/nuisance
theory, registered 18 predictions before training any network (`theory/predictions.md`, frozen at commit
0cf2bc5), and tested them in 1075 held-out runs (seeds 1–5) on 3DShapes.

- **Threshold.** With a covariance penalty, nuisance enters only above λ* = c_p κ/2, with κ = 2α (identity
  predictor) or α(2−α) (learned linear predictor), not λ* ∝ α as first sketched. For any regulariser that
  depends on the embedding only through its covariance, λ*_reg = c_p κ / t*_reg. The constant t*_reg is
  computed numerically, and it is set by the regulariser's slope at a collapsed direction, not by its
  curvature at I.
- **Thresholds in the network.** The four kernel arms show step-like onsets. They order as predicted
  (SIGReg < KSD < sliced KSD < MMD, each CI excludes 1) and do not scale ∝ α: the ratios at α = 0.25 are
  0.50–0.65. With the conv encoder they are always early, at 0.29–0.79× the linear-model value (with a
  linear pixel encoder, late: KSD 1.33×, cov 9.4×). The theory fails at d = 6 ≤ k, where every arm still
  leaks. The same training loop run on the linear model itself also ends with nuisance at d = k (5/6 and 6/6
  seeds at two λ levels; a failed P1 gate that the paper now reports), so this leakage need not come from
  nonlinearity. The thresholds of cov and VICReg are censored by a low-λ collapse regime.
- **Linear leakage is a transient.** At 8000 steps linear leakage falls in all four tested cells, to ≈ 0
  in two. In those two lower-λ cells the network fills the spare directions with nonlinear functions of
  content, the mechanism of Corollary 1(d) (proved zero-cost for cov only). At the higher λ levels kNN
  leakage stays at 0.50 (KSD 3.2) and 0.67 (SIGReg 0.32): there nuisance becomes less linearly accessible,
  not absent.
- **Noise.** Batch-scaled σ raises KSD leakage, as the theory says, but far more than it predicts. The
  predicted fixed-versus-batch ordering is wrong. σ does not raise VICReg's leakage (all 8 contrasts are
  negative, with CIs that include 0).
- **Matched prior.** Orthogonal-fit R² shows no gain. At the uniform law MCC shows one (+0.087 overall,
  +0.25 on the hue pair, where the theory predicted none). At the Laplace law the matched prior lowers both
  metrics.
- **Ranking by frontier.** KSD reaches the most nuisance-free content (content R² 0.727 at leakage ≤ 0.05),
  ahead of MMD (0.63) and then SIGReg ≈ sliced KSD (0.57). VICReg is last. This is not study 1's verdict, and
  at matched content above the thresholds the order changes again.

## Status of each statement (after the P4 proof review)
| statement | status |
|---|---|
| Lemma 1 (invariance cost) | proved |
| T1 / Theorem 1 (cov threshold, m < d−k, d ≤ k) | corrected and proved: λ* = c_pα (identity) / c_pα(2−α)/2 (learned linear predictor); d−k−m directions stay collapsed |
| Corollary 1 (a) probe step, (b) λ* ∝ κ, (c) no threshold at d ≤ k, (d) linearity needed | proved; (d) for the covariance penalty only |
| Heuristic 1 (empty-direction traps) | heuristic, numerically supported; needs a fresh, uncorrelated signal |
| T2 / Theorem 2 (scale reduction λ*_reg = c_pκ/t*_reg) | reduction proved, for kernel arms under Gaussian s and n; constants numerical (independently re-checked for MMD in P4); curvature-at-I proxy refuted; VICReg heuristic (existence of a threshold not proved) |
| T3 / Heuristic 2 (predictor-input noise) | heuristic |
| T4(i) / Proposition 3 (orthogonal invariance) | proved, for σ = 0 or fixed isotropic noise |
| T4(ii) (matched prior identifies up to signed permutation) | Comon 1994, Thm 11, applied; idealised |

## Outcome of each prediction
| # | outcome |
|---|---|
| P1 step-like onset | refuted by the registered criterion (cov and VICReg start above τ); kernel arms step within 0.30–0.60 decades |
| P2 location within ×3 | refuted on the point estimates (KSD 0.32× [0.29, 0.37], sliced KSD 0.29× [0.22, 0.38]; both CIs include 1/3); SIGReg, MMD inside; all early |
| P3 threshold ordering | confirmed for SIGReg < KSD < sliced KSD < MMD; cov ≈ SIGReg inconclusive (censored) |
| P4 α scaling in [α(2−α), 1], not ∝ α | confirmed at α = 0.25 for the four kernel arms (0.50–0.65); at α = 0.5 two point estimates are above 1 (CIs include 1); cov, VICReg censored |
| P5 no threshold at d = 6 | refuted |
| P6 d = 24 ratios | not refuted by the registered criterion; point values miss |
| P7 probe steeper than variance share | inconclusive: probe narrower in 3/6 arms by small margins (was "confirmed (weak), 4/6" before P4 removed a probe-failure level) |
| P8 linear-encoder rung | refuted (cov 9.4× theory, α ratio 0.69); KSD 1.33× (inside) |
| P9 functional form above threshold | cov not refuted by the registered rule (one range met per rung; values below theory); kernel part confirmed |
| P10 KSD/cov ratio > 5 | refuted on the linear rung (3.1); consistent at the conv rung (> 14.8, cov censored) |
| P11 leakage persists to 8000 steps | refuted (leakage falls) |
| P12 what fills spare directions | descriptive: nonlinear content features |
| P13 sign of σ (KSD) | confirmed for leakage; content-share sub-claim not borne out |
| P14 magnitude of σ effect | refuted in magnitude at λ = 0.2; pattern confirmed at λ = 0.1 |
| P15 fixed vs batch σ | refuted (KSD); VICReg inconclusive |
| P16 no orth-fit R² gain from matched prior | not refuted (uniform differences −0.02, +0.03, CIs include 0) |
| P17 no hue-pair MCC gain | refuted |
| P18 theory-guided arm | not tested (E6 cut for compute) |

## Issues fixed in P4
Reports: `study2/review/PROOF_REFEREE.md` and `study2/review/EMPIRICAL_REVIEW.md`; the full list of fixes
is in `study2/gates/GATE_4.md`.
- **Proof review.** No statement was broken. The repairs were one false sentence in a proof (the α = 0 case),
  three unstated assumptions (Gaussian s, n for the kernel arms; a fresh signal in Heuristic 1; no
  batch-scaled noise in Proposition 3), and several overclaims (α(2−α) scaling, Cor. 1(d) applied to the
  kernel arms, the "only on MCC" claim, the existence of a VICReg threshold). All were repaired or downgraded
  in `paper2/paper.qmd` and `theory/theory.md`.
- **Empirical review.** All 60+ recomputed numbers match. Figure 6's 100 runs reproduce exactly, the theory
  checks reproduce, and study 1's files are unmodified. The fixes were:
  - an analysis bug: a probe-failure level drove the P7 verdict. P7 is now inconclusive and P1's MMD
    statistic is corrected;
  - the failed d = k linear-loop check is now reported;
  - "always early" is now scoped to the conv encoder, and the E3 claims to linear leakage;
  - five verdict labels were aligned with the registered criteria (P4, P7, P9, P10, P16);
  - one BibTeX volume (balestriero2022contrastive: 36 → 35);
  - the device statement in REPRODUCE2 now says MPS, not CPU.

## Known gaps
- A linear model of a nonlinear network; one rendered dataset; small models; 5 seeds.
- Proofs are checked numerically and by an independent sub-agent referee, not formally.
- cov and VICReg thresholds are unmeasurable with the registered estimator (collapse regime).
- E6 (theory-guided arm, P18) was cut.
- The theory does not explain why 2000-step training takes the nuisance route first (simplicity bias
  untested), the size of the σ effect, VICReg's insensitivity to σ, kNN (nonlinear) leakage below the
  linear onset, or the hue-pair MCC gain.
- Some literature notes were read from abstracts only (simon2023stepwise, shwartzziv2023information).
- Left as is after review (see GATE_4 decisions):
  - Study 1's frozen note gives KerJEPA's year as 2026, matching its arXiv v2; v1 is from December 2025.
  - The paper prints the cov t* found by bisection (1.952). The closed form, 1.9542, is in `theory/theory.md`.
  - A few qualitative qualifiers in the paper are typed by hand ("two to three times later", "all CIs
    exclude 1"). The reviewer checked each one and found it correct.
