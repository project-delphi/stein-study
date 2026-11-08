# FINAL — Stein regularisation under JEPA bottlenecks (study summary)

## Deliverables
- **Paper:** `paper/paper.pdf`, 14 pages including references and appendix, built from `paper/paper.qmd` (Quarto + typst).
- **Blog:** `blog/index.html`, built from `blog/index.qmd`. It is self-contained and reads only `blog/data/` and `blog/figures/`.
- **Reproduction guide:** `REPRODUCE.md`.
- **Code:** `stein/` and `tests/` (28 pytest tests).
- **Results:** `results/*.parquet`, with the schema in `results/schema.json` and every table in `results/analysis.md`.
- **Related work:** `notes/` (35 verified paper notes and the overlap table) and `paper/refs.bib`.
- **Gates:** `gates/GATE_0.md` … `GATE_4.md`.

## What was done
- **Setup.** 3DShapes view pairs share content (shape, scale, object hue; k = 6 probe dimensions). Their nuisance
  (floor hue, wall hue, orientation) is resampled with probability α.
- **Arms.** Eight JEPA arms: none, VICReg, SIGReg via `lejepa`, sliced KSD, KSD, MMD, EMA, and a rank-k oracle.
- **Axes.** d ∈ {3, 6, 12, 24}, σ ∈ {0, 0.5, 1}, α ∈ {0, 0.5, 1}, and content law ∈ {uniform, Gaussian, Laplace}
  with matched-prior KSD.
- **Further tiers.** A temporal tier (3DShapes clips, EMA-based), a CIFAR-10 tier (augmentation nuisance) and an
  autoencoder tier (β-VAE, KSD-AE, MMD-AE).
- **Protocol.** λ tuned with the same number of configs per arm on 2 validation seeds, then 5 held-out seeds with
  95% bootstrap CIs.
- **Run counts.** 747 training runs, about 28.4 h of compute (192 tuning, 495 held-out, 60 new λ-sensitivity runs;
  counts from `paper/_generated/numbers.yml`).

## Verdicts (details and CIs in the paper and `results/analysis.md`)

| Hypothesis | Verdict | Main reason |
|:--|:--|:--|
| H1: Gaussian targets leak more than VICReg/EMA for d > k | **Refuted under the protocol, λ-dependent** | Leakage rises with λ for every regulariser. Tuned SIGReg leaks more than VICReg at VICReg's least-leaky λ (+0.078 [0.032, 0.120]). |
| H2: σ reduces leakage where d does not | **Refuted (the opposite holds)** | Narrowing d reduces leakage. σ never does (e.g. KSD, d = 12: +0.646 [0.596, 0.687]). |
| H3: a matched prior beats Gaussian targets on identifiability | **Refuted** | Laplace law, matched − Gaussian KSD: −0.496 [−0.558, −0.430]. Full KSD toward N(0, I) is best at every law. |
| H4: EMA differs qualitatively under nuisance | **Refuted** for the direction of the α response | EMA's magnitude differs from VICReg, SIGReg and MMD but lies within their range. Its geometry (low rank) differs. |
| H5 (temporal): EMA alone leaks least; + sliced KSD leaves leakage unchanged | **Refuted on both parts** | EMA − sliced KSD alone: +0.385 [0.262, 0.519]. Adding sliced KSD raises leakage. |
| H6 (autoencoders): AEs leak more than every JEPA arm | **Confirmed** | 21/21 linear contrasts are positive. The matched-prior effect does not carry over. |

All tiers were run; none was cut.

## Independent review (P4)
A fresh sub-agent that had not seen the work checked the PROMPT.md list.
- **It found OK:** number provenance (recomputed from the parquet files), novelty against the overlap table, the
  tiers run, all 35 BibTeX entries, the 14-page render, and a from-scratch reproduction of the Figure 7 temporal
  panel (25/25 runs bitwise identical).
- **It raised no critical issues, 2 major and 8 minor.** All were fixed; see `gates/GATE_4.md`. In short:
  1. The H1 explanation of the tuning rule was wrong for VICReg and MMD. It is corrected, and a contrast in H1's
     direction (tuned SIGReg against VICReg at λ = 1) is now reported.
  2. The H3 conflict with Klindt et al. (their identifiability peaks at the Gaussian law) is now discussed,
     including that probe sets differ by law.
  3. H4 now reports all six arms at α = 0.5, including the EMA magnitude differences.
  4. Borderline contrasts are flagged as weak.
  5. Two overstatements are fixed: d-monotonicity, and "every temporal arm leaks heavily".
  6. The overlap-table column is relabelled "info. bottleneck", and the V-JEPA cell is restored.
  7. Two related-work sentences are softened.
  8. `tools/repro_h5.py` no longer overwrites `results/repro/` when pointed elsewhere.
  9. The reproduction now compares against the paper's own CIs.
  10. The REPRODUCE runtime is now a range.
- **I also fixed:**
  - Figure 1, which was not byte-reproducible (unseeded torch noise);
  - 4 apostrophes after digits that typst rendered as primes.

## Known gaps
- **Scale and data.**
  - The core dataset is rendered (3DShapes at 32×32), with a small discrete factor grid. The CIFAR nuisance is
    defined by the augmentation.
  - The models are small, the runs short, and everything ran on a laptop.
  - 5 held-out seeds make percentile-bootstrap CIs somewhat narrow, so contrasts within about 0.01 of zero are
    read as weak.
- **The tuning rule matters.** λ was tuned once, at the base setting, for content R², and reused across the d, σ
  and α slices. The H1 comparison depends on the selected levels; see the λ-sensitivity slice.
- **The H3 test has low power.** KSD toward a Laplace target has little power at batch size 256, and the trained
  matched embeddings do not reach their prior. The matched prior acts on all 12 dimensions, while the law changes
  2 factors.
- **H2 tests one information bottleneck.** σ is batch-scaled noise on the predictor input. A KL or fixed-scale
  noise bottleneck was not tested.
- **The oracle arm is a diagnostic,** not a method (it needs k).
- **Bibliography.**
  - `kuang2026rectified` is cited as arXiv; an arXiv comment says ICML 2026, but that was not confirmed at the
    venue.
  - Four entries (pu2017vae, vonkugelgen2021self, xu2022kernelised, higgins2017betavae) were confirmed in P4 through
    secondary indexes, because the NeurIPS and OpenReview pages blocked access.
  - The reviewer did not refetch krizhevsky2009learning.
- **Blog.** It needs a local HTTP server to view, because Observable `FileAttachment` does not work from
  `file://`. Its widgets were checked in headless Chrome by the P3 sub-agent and not re-checked in P4.
- **Unused data.** MNIST was preprocessed but not used: the temporal tier uses 3DShapes clips.
- **Author line.** It is "Ravi Kalia". The reproducibility statement discloses that an AI agent ran the
  experiments and drafted the text, and the owner may want to revise it.
- **Logs are not committed.** The P3 reproduction log (`logs/`) and the P4 review scratch (`state/review/`) are
  local only.
