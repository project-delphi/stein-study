# GATE 2 — Sweep and analysis

Status: **in progress**. Session 1 (2026-11-08) handed tuning to the runner (handoff 1 of 2).

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
- None yet.

## Next
- After handoff 1:
  - check `logs/compute-P2-1.log` and `results/tuning_selected.json`, and measure the rate;
  - point `jobs/P2.sh` at `heldout` (out `results/heldout.parquet`); test on a tiny slice; hand off 2.
- After handoff 2: write `stein/analyze.py` (seed-level 95% bootstrap CIs, hypothesis contrasts), give the verdicts,
  and finalise this file.
