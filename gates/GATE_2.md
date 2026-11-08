# GATE 2 — Sweep and analysis

Status: **in progress**. Session 1 handed off tuning (handoff 1, done); session 2 handed off the held-out sweep (handoff 2).

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

## Next
- ~~After handoff 1~~ (done):
  - check `logs/compute-P2-1.log` and `results/tuning_selected.json`, and measure the rate;
  - point `jobs/P2.sh` at `heldout` (out `results/heldout.parquet`); test on a tiny slice; hand off 2.
- `stein/analyze.py` is written and was smoke-tested on the tuning data, with its outputs deleted afterwards. After handoff 2: run `.venv/bin/python -W ignore -m stein.analyze`, which writes `results/analysis.md`, `summary.parquet` and `contrasts.parquet`. Then extend `results/schema.json` with these files, (seed-level 95% bootstrap CIs, hypothesis contrasts), give the verdicts,
  and finalise this file.
