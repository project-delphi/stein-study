# GATE 4 — Independent review

Status: **complete** (2026-11-08). Spend: about $5 of the $53.4 phase allowance. No compute handoff (allowance
0 h); the longest command was the reviewer's 18-minute foreground reproduction.

## Done
- **Independent review.** A fresh general-purpose sub-agent was given only the repo path and the PROMPT.md P4
  checklist. It edited no repo files; its scratch work is in `state/review/` (not committed): recomputation
  scripts, a re-render, and a from-scratch reproduction.
- **What the review found OK:**
  - **Numbers.** It recomputed Tables 2–6 and every H1–H6 contrast it sampled, the λ-sensitivity values,
    the collapse counts, the run totals and the validation numbers from `results/*.parquet`. All means match and
    all CIs agree within bootstrap noise. The paper and the blog take every result through `{{< meta >}}` or
    generated tables, and the remaining literals are design constants.
  - **Novelty.** The three claimed cells match `notes/related_work.md`.
  - **Tiers.** Every claimed tier and arm exists in `results/heldout.parquet` with 5 seeds.
  - **BibTeX.** All 35 entries match on title, authors, year and venue.
  - **Render.** The paper renders at 14 pages and pytest passes (28).
  - **Reproduction.** REPRODUCE §5 reproduced 25/25 runs bitwise into `state/review/`, and 15/15 means are equal.
- **The review raised no critical issues, 2 major and 8 minor.** All 10 are addressed below.

## Review findings and fixes
1. **(major) Wrong explanation of H1 tuning.** The paper said "content R² falls with λ, so the selection rule picks
   low-leakage levels". That is false for VICReg (0.265 / 0.378 / 0.315) and MMD (0.327 / 0.629 / 0.376), and for
   VICReg the rule picked a high-leakage level. → **Fixed** in paper §4.1:
   - the paper now states which level the rule picked for each arm, and that content is not monotone for
     VICReg or MMD;
   - it reports the existing contrast **tuned SIGReg − VICReg at λ = 1: +0.078 [0.032, 0.120]**, which is in
     H1's direction;
   - it says the "less than VICReg" part of the verdict rests partly on VICReg's selected level.

   The blog got the same correction. No analysis code changed: the contrast was already in
   `results/contrasts.parquet`.
2. **(major) The H3 result conflicts with Klindt et al. and the paper did not say so.** → **Fixed** in §4.3:
   - SIGReg, sliced KSD and KSD peak under the Laplace law, while MMD and the oracle peak under the Gaussian
     law; Klindt's sweep peaks at the Gaussian law;
   - the paper lists the differences in setting (continuous latents with d equal to the latent dimension,
     noise-only views; against a reweighted discrete grid, d > k and nuisance);
   - it notes that the probe images are drawn from each law, so scores at different laws are measured on
     different populations (`stein/train.py:204`).
3. **(minor) H4 numbers were cherry-picked.** → **Fixed** in §4.4:
   - all six arms are now given at α = 0.5, and EMA (0.548) is close to VICReg (0.539);
   - the significant EMA interactions with VICReg, SIGReg and MMD are reported;
   - the verdict is kept at "refuted" with the reason given: the direction is shared by all arms and the
     magnitude lies inside the regularisers' range. The blog H4 bullet was fixed the same way.
4. **(minor) The paper's own "weak evidence" rule was not applied.** → **Fixed:** VICReg d 6 − 24 (§4.2) and
   matched-Laplace KSD − SIGReg (§4.3) are now marked weak.
   - The CIFAR contrasts are not cited as significant in the text ("at most a few hundredths"), so no change.
5. **(minor) Overstatements.**
   - "Leakage high for every temporal arm" → **fixed:** the exception, sliced KSD alone at 0.098, is now stated.
   - "Leakage rises with d for every arm" → **fixed** in the Fig. 2 caption and text: every arm leaks more at
     d = 24 than at d = 3, but not monotonically (VICReg is flat from 12 to 24; sliced KSD, KSD and MMD stay
     near 0 up to d = 12).
6. **(minor) Overlap-table column mislabelled.** → **Fixed:** "injected noise" is now "info. bottleneck",
   matching `notes/related_work.md`, and V-JEPA's "implicit (narrow predictor)" is restored.
7. **(minor) Two related-work statements were weakly supported.**
   - "agrees with KerJEPA's use of full KSD" → "consistent with KerJEPA including unsliced KSD among its variants";
   - "these analyses assume views differ only by noise" → now attributed to Klindt's identifiability analysis,
     plus "none of these JEPA works has ground-truth nuisance in its views".
8. **(minor) `tools/repro_h5.py` hardcoded its output paths.** → **Fixed:**
   - outputs now go to the directory of the input parquet;
   - the docstring now says `state/runs_repro`;
   - REPRODUCE §5 explains how to reproduce without touching `results/repro/`.
9. **(minor) The reproduction compared against its own bootstrap.** → **Fixed:** the "reported" column is now read
   from `results/summary.parquet`, so it is exactly the paper's CIs. The script also counts exactly equal means.
   - The regenerated `results/repro/h5_compare.md` reads: 15/15 means inside the CI, and 15/15 exactly equal.
   - The P3 runs were reused for this; the sweep itself did not change.
10. **(minor) Runtime.** REPRODUCE §5 now says "about 15–18 minutes": 923 s in P3, and about 18 min in the review
    (43 s/run effective, from `state/review/repro_sweep.log`).

## Found by me during the fixes
- **Figure 1 was not byte-reproducible.** `fig_views` drew pixel noise from torch's unseeded global RNG.
  → `stein/report.py` now seeds torch. Two regenerations gave an identical embedded image. Other figures differ
  only in SVG metadata (date, clip ids).
- **Pandoc/typst rendered "H1's" and "view 1's" as a prime (′)** after a digit, in 4 places in the paper. → I
  rephrased all 4, and the PDF now has 0 primes. The blog HTML was unaffected.
- **My own blog edit was wrong.** It said "leakiest but one", but VICReg's middle level leaks 0.673 against 0.671
  at its highest. → Corrected, with the numbers taken from numbers.yml.

## Decisions
- **H1 verdict stays "refuted under the protocol, and dependent on λ"**, now with the explicit SIGReg − VICReg
  (λ = 1) caveat. The pre-registered protocol compares tuned arms, and VICReg's tuned level is part of that
  protocol.
- **H4 verdict stays "refuted"** rather than "inconclusive". H4 is about a *qualitative* difference: the
  direction of the α response is shared, and EMA's magnitude lies inside the regularisers' range. The magnitude
  differences are now reported.
- **The abstract** now says "when each is tuned for content" instead of "at content-tuned strength".
- **kuang2026rectified stays an arXiv `@misc`.** An arXiv comment says ICML 2026, but the venue's proceedings
  were not checked.
- **GATE_2 is left as the historical record.** The paper and blog are now the corrected statements.

## Final checks (after all fixes)
- `python -m stein.report` regenerated the outputs. `numbers.yml` and the generated tables are unchanged.
- `quarto render` gives `paper/paper.pdf` at **14 pages**, with 0 unresolved shortcodes in the PDF and the HTML.
- pytest: 28 passed.
- Every newly cited number, checked against `paper/_generated/numbers.yml`:
  - +0.078, −0.170, −0.177, −0.345, −0.122;
  - 0.539, 0.338, 0.098;
  - 0.265 / 0.378 / 0.315 and 0.327 / 0.629 / 0.376.

## Problems
- None blocking. Known gaps are listed in `FINAL.md`.

## Next
- None. The study is finished.
