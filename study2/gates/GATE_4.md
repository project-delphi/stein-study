# GATE 4 — Independent review (study 2) — FINAL

## Done
- Commit before study 2: c3dd0d8. P0 commit (predictions frozen): 0cf2bc5.
- **Two independent reviewers.** Each was a fresh general-purpose sub-agent given only the repo path and its
  checklist.
  - The first session ended while both were running in the background, and the next session resumed them.
  - The empirical reviewer died a second time, so a fresh one was run in the foreground. Its predecessor's
    /tmp scratch logs were available to it, to be confirmed rather than trusted.
- **Proof referee:** `study2/review/PROOF_REFEREE.md`, with its own scripts and outputs in
  `study2/review/proof_referee/`.
  - It found no counterexample to any statement called a theorem, proposition or lemma.
  - It checked them independently: L-BFGS over (A, B, W) in 120 settings × 12 starts, with none below the
    closed form, and 20,000 random (P, Q).
  - Its MMD t* check, which it had not run, was run in P4: 0.004882 / 0.002311 (d 12 / 24) against
    `theory_t2.json` 0.004873 / 0.002309, within 0.2 %.
- **Empirical reviewer:** `study2/review/EMPIRICAL_REVIEW.md`.
  - More than 60 numbers recomputed; all match.
  - `predictions.md` is unchanged since 0cf2bc5, and all 18 outcomes are reported.
  - `git diff c3dd0d8` over study 1's paths is empty.
  - check_t1–t4 and validate2 reproduce, except `runtime_s`.
  - All 100 Figure 6 runs were rerun into an empty `STEIN_RUNS`. They match in all 12,400 values, and
    regenerating gives an identical `fig6_e4.svg`.
- **A third fresh sub-agent verified the fixes** against both reports. It found no unresolved finding and
  listed 11 residual wording or code issues, all fixed (below).
- **Final state.** `paper2/paper.pdf` renders with no warnings to **12 pages**, and all 142 meta keys resolve.
  The 40 tests pass. `FINAL2.md` is written.

## Fixes
### Proof referee (F1–F13), in `paper2/paper.qmd` and `theory/theory.md`
| # | finding | fix |
|---|---|---|
| F1 | Cor. 1 proof: "at α = 0, any split with P+Q = I is optimal" is false for d > k+m | min(d, k+m) unit eigenvalues, rest 0; min J = λ(d−k−m) when d > k+m; α = 0 is taken as a limit |
| F2 | Theorem 2's kernel arms need Gaussian s and n | Assumption stated; Table 1 constants are approximations for discrete 3DShapes |
| F3 | Cor. 1(d), proved for cov, was invoked as zero-cost for KSD and SIGReg | "mechanism of Cor. 1(d), proved free for cov only" in the abstract, E3 and discussion |
| F4 | α = 0.5 ratios above 1 were called "allowed by the trap mechanism" | Point estimates are above the trap bound of 1; the CIs include 1 |
| F5 | "scale as α(2−α)" overclaims | "not ∝ α"; ratios lie above α(2−α), near or above 1 |
| F6 | "only on MCC" and "as the theory requires" | Qualified to the same-orbit case; Laplace lowers both metrics; "requires" → "allows" |
| F7 | Heuristic 1 needs a fresh signal; c_p/2 applies to cov only | Condition added; "up to about c_p/2 for cov (more if c(u) > 1)" |
| F8 | Prop. 3 needs no noise or fixed isotropic noise; the sliced arms are invariant in expectation | Both added to the statement and the proof |
| F9 | Thm 1(a) proof for d ≤ k; the min-J term can be negative | Proof written out; (d−k−r)₊ |
| F10 | The j-factor feature cost holds for pure interactions only | Corrected in theory.md |
| F11 | VICReg threshold existence | "Theorem 2's reduction applies; existence not proved" |
| F12 | Numbers disagree across files | Reconciled; cov t* closed form 1.9542 re-derived and noted in theory.md |
| F13 | Three wording issues | Fixed |

### Empirical reviewer (M1–M4, m1–m12)
- **M1, analysis bug.** At MMD λ = 2.5, a probe failure (mean leakage −7.74) set the path minimum in P7's
  `rise_width`.
  - `stein/verdicts2.py` now drops levels with mean leakage below −0.5 in P7 and P1. That is the filter
    `report2` already used for the figures.
  - MMD's probe width goes from 0.25 to 0.93 decades, the probe is narrower in 3/6 arms instead of 4/6, and
    P7 changes from "confirmed (weak)" to **inconclusive**. P1's MMD `leak_lowest` goes from −7.74 to 0.004.
  - The Figure 2 caption now uses the new key `mmd_probe_fail`.
- **M2.** The failed d = k gate of the linear model in the real training loop is now reported. The loop
  ends with nuisance in 5/6 and 6/6 seeds, with leakage 0.19–0.36. The P5 explanation now reads "need not
  come from nonlinearity".
- **M3.** "Always early" is scoped to the conv encoder; the linear pixel encoder is late.
- **M4.** The claims are now about *linear* leakage being a transient. The kNN leakage at 8000 steps (0.50,
  0.67) is reported for the higher-λ cells, and "replaces nuisance" is limited to the lower-λ cells.
- **m1.** P1's margin is stated (≤ 0.002, generated as `p1_margin`), on near-collapsed embeddings.
- **m2.** P2 ratio CIs are added to `verdicts.json` (`ratio_ci`), the text and the table; both include 1/3.
- **m3 / m4: verdict labels** in `report2.py` → `tab_verdicts.md`:

  | prediction | new label |
  |---|---|
  | P4 | confirmed at α = 0.25; 2 above 1 at α = 0.5 |
  | P8 | refuted (cov location and α ratio), with the α ratios in the statistic |
  | P9 | cov not refuted (one range met per rung) |
  | P16 | not refuted |

- **m5.** P10 is consistent at the conv rung (> 14.8, cov censored), reported beside the refutation on the
  linear rung.
- **m6.** VICReg σ: all 8 contrasts are negative (−0.16 to −0.02), with CIs that include 0.
- **m7.** P14's threshold ratios are stated as not measured.
- **m8.** The T4 invariance sentence separates KSD (floating-point error) from the |z| bound.
- **m9.** balestriero2022contrastive volume 36 → 35 (in the note, then `build_bib.py`).
- **m10.** REPRODUCE2 now says training is deterministic on MPS (tested) and that other devices are
  untested. It also records the reviewer's reproduction.
- **m11.** Re-rendered; 12 pages.
- **m12.** The flagged counts (P7, P1 margin, VICReg contrasts, d = k seeds) are now generated. The remaining
  qualitative qualifiers were checked correct by the reviewer.

## Decisions
- D1. The reports are kept in the repo (`study2/review/`). The proof referee could not write to /tmp, so its
  report was saved from its final message.
- D2. F11's "R lower semicontinuous and coercive" is not added as a formal assumption. Every R used is
  continuous and coercive at the scales involved, and the existence of t* was already labelled numerical.
- D3. The paper keeps the bisection value of cov t* (1.952), which is what the scripts compute; the exact
  closed form is in theory.md. λ* = 0.0427 either way.
- D4. KerJEPA's year (2026) is left unchanged. The note belongs to study 1, which is frozen, and the year
  matches the arXiv record for v2. The `@misc` types of two ICML papers are also left as is, being cosmetic.
- D5. The post-hoc analysis changes (the probe-failure filter in P1/P7 and the P2 ratio CI) fix a bug found
  in review. They are not a change of estimator: the registered estimator is unchanged, and only a level
  that the paper already called a probe failure, and excluded from Figures 2 and 4, is now excluded from P7
  too. GATE_2 (a P2 record) is left as written. The changed verdicts are recorded here and in FINAL2.md.
- D6. The empirical reviewer's reruns rewrote `runtime_s` in `results2/theory_t*.json` and
  `validation2.json`. No other field changed, so these files are left as rewritten.

## Problems
- The page budget is exhausted (exactly 12 pages).
- Background sub-agents do not survive the end of a session. Reviews must run in the foreground.

## Next
- None. Study 2 is complete.
