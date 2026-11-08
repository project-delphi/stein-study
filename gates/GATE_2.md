# GATE 2 — Sweep and analysis

Status: **complete** (2026-11-08). P2 used 2 handoffs (tuning 1.4 h, held-out 3.0 h) and one 25-min foreground robustness slice. All tiers were run; none was cut.

## Done
- Read GATE_0/GATE_1. Reviewed stein/train.py, stein/sweep.py, results/schema.json.
- σ mini-pilot (`configs/pilot/p2_sigma.yaml` → `results/pilot_p2_sigma.parquet`, log `logs/pilot_p2_sigma.log`):
  - 14 runs in 260 s wall at 6 MPS workers, so **18.6 s/run effective**.
  - Train time per run at 6 workers: shapes 55–96 s, CIFAR 100–112 s.
  - Seed 0, single runs. These are not results.
    - σ = 2 leaves EMA with almost no content (kNN content R² 0.007, embedding sd 118: EMA grows z).
    - SIGReg, VICReg and MMD still learn something at σ = 2.
- `stein/grids.py`: `tune()` (192 runs), `heldout()` (needs the selection), `tune_tiny()` (smoke test),
  `slice_membership()`.
- `stein/select_tuning.py` writes `results/tuning_selected.json` and `results/tuning_table.md`.
- `jobs/P2.sh` (handoff 1) runs `tune` and then `select_tuning`.
  - Tested on `tune_tiny`: 6 runs at 50 steps, one per tier, 0 failures.
  - The smoke-test runs and their outputs were deleted afterwards.

- **Handoff 1 (tuning) finished.** Exit 0 after 1.4 h; 192/192 runs, 0 failed (`logs/compute-P2-1.log`).
  - `results/tuning.parquet`, `results/tuning_selected.json` (32/32 slots), `results/tuning_table.md`.
  - Measured rate: 26 s/run effective. Wall clock ≈ 1.04 × (sum of per-run times / 6).
  - Per-run time by arm (`results/tuning.parquet`):
    - 3DShapes: 66–110 s, sliced KSD 227 s;
    - temporal: 136–183 s, sliced-KSD arms 314–330 s;
    - CIFAR: 108–197 s, sliced KSD 310 s;
    - VAE: 93–108 s.
- **Selected levels** (`results/tuning_table.md`):
  - 3DShapes base: VICReg 5, SIGReg 0.05, sliced KSD 1, KSD 0.2, MMD 10, EMA τ 0.999, oracle 0.1.
  - Matched priors: KSD-uniform 0.2, KSD-Laplace 1, sliced-KSD-uniform 5, sliced-KSD-Laplace 5.
  - Laplace law: SIGReg 0.01, sliced KSD 1, KSD 0.2, MMD 10, oracle 0.02.
  - Temporal: EMA τ 0.99, ema+sksd 5, ema+sigreg 0.05, sliced KSD 1, VICReg 1.
  - CIFAR: VICReg 5, SIGReg 0.05, sliced KSD 5, KSD 0.2, MMD 10, EMA τ 0.999.
  - VAE: β 0.2, KSD-AE 10, MMD-AE 20; Laplace law: KSD-AE (Laplace prior) 2, KSD-AE (Gaussian prior) 2.
- **Handoff 2.** `jobs/P2.sh` now runs `python -m stein.sweep heldout` → `results/heldout.parquet`.
  - 495 runs (99 configs × 5 seeds).
  - Estimate from the measured per-arm times, cumulative in priority order: H1 0.84 h, H4 1.18, H2 1.85, H3 2.31,
    H5 2.59, CIFAR 2.87, H6 **3.0 h**.
  - Allowance: 4.6 h left, which is 3.2 h after the 30% margin, so all tiers are kept.
  - Smoke-tested on `heldout_tiny`: 7 runs, one per block, at 50 steps, 0 failures. The outputs were deleted.

- **Handoff 2 (held-out) finished.** Exit 0 after 3.0 h; 495/495 runs, 0 failed, 0 non-finite losses
  (`logs/compute-P2-2.log`, `results/heldout.parquet`).
- **λ-sensitivity robustness slice** (foreground, 25 min; `python -m stein.sweep lam_sens --workers 6 --out
  results/lam_sens.parquet`, log `logs/lam_sens.log`):
  - all 3 tuning levels of VICReg, SIGReg, sliced KSD, KSD, MMD and oracle at the base setting, on held-out seeds
    1–5;
  - 90 configs × seeds, 60 of them new; 0 failed.
  - Added because tuning showed leakage tracking λ, which makes H1 depend on λ.
- **Analysis.** `.venv/bin/python -W ignore -m stein.analyze` writes:
  - `results/analysis.md` (all tables);
  - `results/summary.parquet`, `results/contrasts.parquet` (381 contrasts);
  - `results/tuning_summary.parquet`, `results/lam_sens_summary.parquet`.
  - CIs are 95% percentile bootstrap over seeds (10000 resamples). Contrasts resample each group's seeds
    independently.
  - `results/schema.json` now documents every results file.
  - `pytest`: 28 passed.

## Hypothesis verdicts
Every number below comes from `results/analysis.md`. Values are means over held-out seeds 1–5 with [95% bootstrap
CI]. The primary leakage metric is linear `leak_r2`; kNN `leak_knn_r2` is secondary. k = 6.

### H1 — for d > k, isotropic-Gaussian regularisers leak more nuisance than VICReg and EMA: **REFUTED** under the protocol (content-tuned λ), and it depends on λ.
- **At the tuned λ, d = 12** (linear leakage):
  - SIGReg 0.250 [0.218, 0.281], sliced KSD 0.019 [0.008, 0.029], KSD 0.002 [−0.001, 0.005], MMD 0.012
    [−0.000, 0.026];
  - VICReg 0.673 [0.542, 0.772], EMA 0.215 [0.137, 0.283].
- **Against VICReg:** all 8 Gaussian-arm − VICReg contrasts at d ∈ {12, 24} are negative, with CIs that exclude 0.
- **Against EMA:**
  - at d = 12: sliced KSD, KSD and MMD are lower than EMA; SIGReg is not different (+0.035 [−0.041, 0.113]);
  - at d = 24: sliced KSD and MMD are lower than EMA; SIGReg (−0.057 [−0.169, 0.056]) and KSD (−0.005 [−0.146,
    0.134]) are not different.
- **Exception:** SIGReg's kNN leakage is above EMA's at every d (d = 12: +0.158 [0.094, 0.217]).
- **Leakage grows with d for every arm, not only the Gaussian ones** (linear, d = 3 → 24):
  - VICReg 0.23 → 0.67; SIGReg 0.02 → 0.45; KSD 0.00 → 0.50; EMA 0.03 → 0.50; oracle 0.00 → 0.69;
  - sliced KSD 0.00 → 0.135 and MMD 0.00 → 0.057 stay lowest at d = 24.
- **λ dependence** (`lam_sens`, held-out seeds):
  - highest-λ − lowest-λ leakage is positive with the CI excluding 0 for **every** regulariser, VICReg included:
    VICReg +0.50, SIGReg +0.37, sliced KSD +0.52, KSD +0.53, MMD +0.33, oracle +0.49;
  - at λ ×5 above its content-optimal level, every Gaussian arm leaks **more** than VICReg's least-leaky level
    (λ = 1, 0.172), and more than EMA except MMD at λ = 50 (+0.135 [−0.003, 0.271]) (KSD λ = 1: +0.210 [0.146, 0.266] vs VICReg; +0.166 [0.080, 0.254] vs EMA).
- **Reading:** more distributional pressure buys nuisance leakage. That holds for every regulariser, so it is not
  specific to isotropic-Gaussian targets. The tuned λ, not the target law, decides the H1 comparison.

### H2 — the information bottleneck (σ) reduces leakage where the dimensional one (d) does not: **REFUTED**. The opposite holds.
- **Narrowing d reduces linear leakage for all 6 arms** (σ = 0, d = 6 − d = 24, CI excludes 0):
  - EMA −0.43, VICReg −0.18, SIGReg −0.40, sliced KSD −0.135, KSD −0.50, MMD −0.05.
- **Adding σ never reduces linear leakage.**
  - σ = 1 − σ = 0 is positive for EMA, SIGReg, sliced KSD, KSD and MMD at both d. For example:
    - KSD, d = 12: +0.646 [0.596, 0.687];
    - MMD, d = 12: +0.405 [0.347, 0.469].
  - VICReg is unchanged (d = 12: −0.103 [−0.243, 0.050]).
  - σ also lowers content R² for every arm.
- **Only exception:** EMA's **kNN** leakage falls with σ (d = 12, σ = 1: −0.107 [−0.153, −0.060]), but its kNN
  content falls more (0.785 → 0.244). Its embedding is degrading, not filtering nuisance.
- **Caveat:** σ is noise on the predictor input, scaled to the batch sd of z. That is one choice of information
  bottleneck among several.

### H3 — with non-Gaussian content laws, KSD with a matched prior beats Gaussian-target regularisers on identifiability (`ident_orth_r2`): **REFUTED**.
- **Uniform law:** matched-uniform KSD 0.554 [0.501, 0.598] vs Gaussian-target KSD 0.576 [0.543, 0.595], difference
  −0.022 [−0.082, 0.036].
  - It does beat SIGReg, sliced KSD, MMD and oracle, but so does Gaussian KSD. The advantage belongs to full KSD,
    not to the matched prior.
- **Laplace law:** matched-Laplace KSD 0.276 [0.235, 0.317] vs Gaussian KSD 0.772 [0.717, 0.813], difference
  −0.496 [−0.558, −0.430].
  - It is also below SIGReg (−0.082) and sliced KSD (−0.167), and not different from MMD or the oracle.
- **Matched sliced KSD** (uniform and Laplace) is below every Gaussian-target arm at both laws.
- **Full KSD toward N(0, I) gives the best identifiability at every law:** uniform 0.576, Gaussian 0.701,
  Laplace 0.772.
- **Caveats:**
  - Laplace-prior KSD has low power at n = 256 (GATE_1);
  - the trained matched-Laplace embeddings do not reach their prior: NP-KSD 1.14 (KSD) and 7.42 (sliced KSD) vs
    ≈ 0.02 for Gaussian KSD toward N(0, I);
  - the matched priors act on all d = 12 dimensions, while the law changes only 2 content factors.

### H4 — EMA/stop-gradient behaves qualitatively differently from distributional regularisers under nuisance: **REFUTED** for the leakage response. Its geometry does differ.
- **Same direction for every arm:** raising α from 0 to 1 lowers or leaves leakage unchanged and raises content.
  - EMA leakage −0.458 [−0.537, −0.384];
  - KSD −0.410, sliced KSD −0.521, MMD −0.336, SIGReg −0.113;
  - VICReg −0.041 [−0.180, 0.085], not different from 0.
- **EMA's response sits inside the range of the regularisers** (difference-in-differences, linear):
  - not different from KSD (−0.048 [−0.161, 0.055]) or sliced KSD (+0.063 [−0.036, 0.155]);
  - larger than VICReg (−0.417), SIGReg (−0.345) and MMD (−0.122).
- **At the middle level α = 0.5:** EMA leaks 0.548, while sliced KSD, KSD and MMD are ≈ 0.001–0.006.
- **What is different is geometry:** EMA's embedding sd is about 0.1, and its effective rank is 4.0 at α = 1, against
  5.7–11.0 for the regularisers.

### H5 (temporal tier) — EMA alone leaks less than any distributional regulariser alone, and adding sliced KSD to EMA leaves leakage unchanged: **REFUTED** on both parts.
- **Linear leakage:** EMA 0.484 [0.381, 0.601], sliced KSD alone 0.098 [0.033, 0.164], VICReg alone 0.413
  [0.307, 0.518].
  - EMA − sliced KSD = +0.385 [0.262, 0.519]: EMA leaks *more*.
  - EMA − VICReg = +0.071 [−0.081, 0.230], not different. On kNN, EMA leaks more than VICReg (+0.212).
- **Adding sliced KSD to EMA raises leakage:** +0.229 [0.110, 0.337]. EMA + SIGReg: +0.121 [−0.005, 0.236].

### H6 (VAE tier) — at matched d, reconstruction-based models leak more than every JEPA arm: **CONFIRMED**.
- **VAE-tier linear leakage:** β-VAE 0.809, KSD-AE 0.912, MMD-AE 0.855.
- **All 21 contrasts** (3 autoencoders × 7 JEPA arms at d = 12, linear) are positive with the CI excluding 0. So
  are all 21 kNN contrasts.
  - Smallest gap: β-VAE − VICReg = +0.136 [0.028, 0.267].
- **H3's matched-prior carry-over: it carries over as a null/negative result.**
  - At the Laplace law, the Laplace-prior KSD-AE has **lower** identifiability than the Gaussian-prior KSD-AE:
    −0.121 [−0.204, −0.036].
  - Content R² does not differ: −0.011 [−0.081, 0.054].

### Natural-image tier (CIFAR-10, no hypothesis; one slice)
- **Class accuracy:** KSD 0.357, sliced KSD 0.348, MMD 0.341, VICReg 0.333, SIGReg 0.332, EMA 0.299, none 0.115.
- **Augmentation leakage is small for every arm,** 0.018–0.049. Gaussian arms vs VICReg range from −0.004 to
  +0.016; every arm is below EMA (0.049).
- On natural images, isotropic regularisers do not leak augmentation parameters linearly to any notable degree.

## Key tables
`results/analysis.md` has all of them (H1 d × arm, H2 σ × d, H3 law × arm, H4 α × arm, H5, CIFAR, H6, λ
sensitivity, collapse).

## Surprises
1. **σ is not an information bottleneck for nuisance here.** Noise on the predictor input raised leakage for every
   distributional arm, while a smaller d lowered it (H2 reversed).
2. **Full KSD with an N(0, I) target is the strongest arm on 3DShapes.** It has the best content R² (0.727) and
   identifiability (0.576) at d = 12, and its best identifiability is under a *Laplace* law (0.772). Matched
   non-Gaussian priors hurt.
3. **Regularisation strength, not target law, governs leakage.** Every regulariser, VICReg included, leaks more as
   λ grows (λ-sensitivity slice).
4. **In the temporal tier, adding sliced KSD to EMA increases leakage,** and sliced KSD alone leaks least.
5. **The oracle (rank-k target) leaks heavily for d > k:** 0.451 at d = 12 and 0.685 at d = 24. Shrinking the d − k
   extra dimensions to sd 0.3 does not remove nuisance from them, because linear probes are scale-invariant.

## Decisions
- **Seeds.**
  - Tuning seeds are {100, 101}.
  - **Held-out seeds are {1, 2, 3, 4, 5}**, not {0..4} as GATE_1 suggested. All P1 pilots used seed 0, and P1
    picked its defaults (steps, width, EMA recipe) on them. Excluding seed 0 keeps the held-out runs clean of that
    selection, and stops pilot run directories being reused through a hash collision.
- **Matched tuning.**
  - Every tuned arm gets exactly 3 configs: λ × {1/5, 1, 5} around the P1 centres (VICReg 5, SIGReg 0.05, sliced
    KSD 5, KSD 1, MMD 10, oracle 0.1; VAE: β 1, KSD-AE 10, MMD-AE 100), or EMA τ ∈ {0.99, 0.996, 0.999}.
  - "none" has nothing to tune.
  - The ema+X temporal arms tune λ and keep τ = 0.999 (the P1 default) to avoid a dependency between arms.
- **Selection criterion.** The highest mean validation **linear content R²** (CIFAR: class accuracy) over the 2
  tuning seeds.
  - A level is not eligible if it has any non-finite loss.
  - Leakage and identifiability are never used, so the selection cannot favour any hypothesis's outcome.
- **Where tuning happens (tuning slots).**
  - The core arms are tuned at the base setting: 3DShapes, d = 12, α = 1, σ = 0, uniform law.
  - The H1, H2 and H4 slices reuse those levels.
  - H3:
    - the matched-prior arms (KSD and sliced KSD with a uniform prior at the uniform law, and with a Laplace prior
      at the Laplace law) are tuned at their own law;
    - at the Laplace law, every H3 arm (SIGReg, sliced KSD, KSD, MMD, oracle) is re-tuned, so matched and
      Gaussian-target arms are tuned under the same conditions;
    - at the Gaussian law every arm targets N(0, I) (KSD *is* the matched arm there), so all use the base levels.
  - Each further tier is tuned at its own base setting:
    - temporal: T = 3, α = 1, d = 12;
    - CIFAR: α = 1, d = 12;
    - VAE: d = 12, uniform law;
    - VAE, for H6's carry-over of H3: a KSD autoencoder with a Laplace prior vs one with a Gaussian prior, at the
      Laplace law.
- **The oracle arm is dropped on CIFAR.** The oracle needs the known content rank k, and CIFAR has no known k.
- **σ levels** (mid, high) = **{0.5, 1.0}**, so per-dimension SNR = 4 and 1.
  - σ = 2 left EMA with no content in the mini-pilot, so 1.0 is the "high" level that still trains.
- **The held-out grid** (`grids.heldout`, 5 seeds each, deduplicated by config hash) runs in priority order, so a
  cut-off keeps the cut order (VAE first, then CIFAR):
  - H1: d ∈ {3, 6, 12, 24} × 8 arms;
  - H4: α ∈ {0, 0.5, 1} × {EMA, VICReg, SIGReg, sliced KSD, KSD, MMD};
  - H2: σ ∈ {0, 0.5, 1} × d ∈ {12, 24} × the same 6 arms;
  - H3: law ∈ {uniform, Gaussian, Laplace} × {SIGReg, sliced KSD, KSD, MMD, oracle}, plus matched-prior KSD and
    sliced KSD (uniform and Laplace);
  - H5: the 5 temporal arms;
  - CIFAR: 7 arms;
  - H6: 3 VAE arms, plus the 2 Laplace-law KSD-AEs.
  - Total: 101 configs, about 505 runs.
- **Primary leakage metric:** linear probe R² (`leak_r2`), as the spec defines it. kNN leakage (`leak_knn_r2`) is
  reported beside it as secondary, because P1 found nonlinear-only leakage.

## Sizing
- Tuning: 192 runs.
  - The estimate is the sum of per-run (train + eval) times at 6 workers / 6 × 1.15 overhead, which gives about
    **70 min**.
  - That uses 75 s per shapes run, 195 s per sliced-KSD run, 115 s per CIFAR run, 160 s per temporal run and
    85 s per VAE run.
- Held-out: about 505 runs ≈ 2.7 h at the same rates.
- Total ≈ 3.9 h, against the 4.2 h available (6 h minus the 30% margin). Handoff 2 is re-sized from the rate
  measured in handoff 1.

## Problems
- **Several selections sit at the lowest λ of their grid**:
  - 3DShapes: sliced KSD 1, KSD 0.2, KSD-uniform 0.2;
  - Laplace law: SIGReg 0.01, oracle 0.02;
  - VAE: β 0.2, MMD-AE 20, and both Laplace-law KSD-AEs at 2.
  - A lower λ might score higher still. There was no budget to extend the grids, because matched tuning would need
    every arm extended, so this is reported as a limitation.
- **The tuning runs show a strong λ–leakage trend** (validation seeds). Higher λ gives more linear nuisance leakage
  for every Gaussian-target arm on 3DShapes, for example:
  - KSD: 0.003 → 0.58 → 0.68;
  - sliced KSD: 0.03 → 0.53 → 0.56;
  - SIGReg: 0.01 → 0.29 → 0.41.
  - Because the selection favours content R², it tends to pick the low-λ levels with low leakage. The tuned λ
    therefore matters for H1, and the analysis must report this trend from `results/tuning.parquet`. It is
    validation data, not held-out data.
- **The EMA arm's seeds disagree at τ = 0.99 and τ = 0.999** (content R² 0.60 vs 0.18, and 0.63 vs 0.38). EMA is
  the least stable arm.

- **Collapse** (`analysis.md`, collapse section):
  - "none" collapses everywhere: embedding sd < 0.01 in 100% of seeds;
  - at d = 3, EMA collapsed in 3/5 seeds and MMD in 2/5, so the d = 3 comparisons for those arms mix collapsed
    and non-collapsed runs;
  - EMA with σ > 0 has an unstable loss (CV 0.58–0.92 over the last 20% of steps).
- **λ was tuned only at the base setting (d = 12, σ = 0, α = 1).** The H1, H2 and H4 slices reuse it, so part of the
  d and σ effects may come from a λ that is no longer optimal there. This is a stated protocol limitation.
- Different arms' runs are **not paired** across seeds, so contrasts use independent bootstrap resampling.
- The CIs come from 5 seeds and the percentile bootstrap, which is somewhat narrow at n = 5. Read borderline
  contrasts (CI edge within about 0.01 of 0) as weak.

## Next (P3)
- Build the paper and blog from `results/heldout.parquet`, `results/lam_sens.parquet`, `results/summary.parquet`
  and `results/contrasts.parquet` (schema in `results/schema.json`); regenerate the tables with `stein.analyze`.
- Suggested figures:
  1. leakage vs d per arm (H1);
  2. λ sensitivity (leakage vs λ);
  3. σ × d leakage (H2);
  4. identifiability by law × arm (H3);
  5. leakage vs α (H4);
  6. temporal tier and VAE vs JEPA leakage (H5/H6).
