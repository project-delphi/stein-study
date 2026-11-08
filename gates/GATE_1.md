# GATE 1 — Implementation and validation

Status: **complete** (2026-11-08). Spend: about $7.5 of the $30.7 phase allowance. No compute handoff was used.

## Exit criteria
- [x] `stein/` package: plain PyTorch, YAML configs, one entry point (`python -m stein.train <cfg.yaml> k=v ...`),
  grid runner `python -m stein.sweep`. Runs are keyed by a config hash (`runs/<hash>/result.json`), finished runs
  are skipped, seeds are fixed, and `requirements.txt` is pinned (`uv pip freeze`, lejepa at c293d291).
- [x] Unit tests: `.venv/bin/python -m pytest -q tests`, **28 passed**. There is at least one test per loss, plus
  gradchecks, sampler tests and probe tests.
- [x] Every validation gate passes. Numbers are in `results/validation.json`, produced by
  `python -m stein.validate`, and summarised below.
- [x] Pilot: at least one run per arm for every tier kept, timed on CPU and MPS. The device is chosen and P2 is
  estimated (below).

## Validation gates (results/validation.json; d = 12, n = 256, mean of 5 draws)
| regulariser | N(0,I) | collapsed (1+0.01N) | rank-1 | anisotropic (sd 3/1/0.1) | shifted N(0.5,I) |
|---|---|---|---|---|---|
| VICReg (var + 0.04 cov) | 0.018 | 0.986 | 0.716 | 0.248 | 0.020 |
| SIGReg (lejepa, EP·N) | 0.886 | 278.1 | 80.8 | 33.1 | 31.2 |
| sliced KSD (IMQ) | −0.0013 | 2.10 | 0.663 | 0.167 | 0.159 |
| full KSD (IMQ) | −0.0007 | 23.9 | 4.73 | 0.365 | 0.629 |
| MMD (Gaussian) | 0.0001 | 0.597 | 0.213 | 0.060 | 0.050 |
| oracle (rank-6 target, eps 0.3) | 4.99 (not its target) | 757.8 | 11.4 | 1.40 | 43.9 |

- **Values at the target law:**
  - KSD, sliced KSD and MMD are unbiased U-statistics, so they sit ≈ 0 at N(0, I).
  - SIGReg's statistic (Epps–Pulley × N) has a positive expectation under the null, 0.89 here. That is small next
    to 33–278 for the alternatives.
  - The oracle's own null is its rank-6 target, where it gives −0.004.
  - VICReg is mean-shift invariant by design (shift 0.020 vs null 0.018). This is a known property, not a defect.
- **Matched priors (full KSD):**
  - Uniform prior, scored with the GN(p=8) surrogate: exact-uniform sample 0.0075 vs N(0, I) 3.40.
  - Laplace prior: Laplace sample 0.0019 vs N(0, I) 0.0040. The test has **little power at n=256** (see Problems).
- **SIGReg matches lejepa** on the same batch and directions:
  - relative difference 0.0 at (d=12, n=256) and (d=64, n=512);
  - gradients match too (`tests/test_losses.py::test_sigreg_matches_lejepa`).
- **Gradient checks:**
  - autograd gradcheck in float64 passes for KSD (Gaussian, Laplace, uniform and rank-k scores), sliced KSD, MMD,
    VICReg and SIGReg;
  - the closed-form IMQ Stein kernel equals an autograd-built Stein kernel (all three priors);
  - 1-D sliced KSD equals full KSD.
- **Pair sampler:**
  - content is identical across views at every α;
  - α = 0: the views are pixel-identical before view noise (max diff 0.0);
  - α = 0.5: cross-view nuisance correlation 0.50–0.51;
  - **α = 1: max |cross-view corr| = 0.0099** over the 5×5 nuisance codes (n = 40000);
  - the test split is never touched by training pairs;
  - the latent-law frequencies match their targets (test);
  - clips keep content constant, and nuisance moves ±1 per frame;
  - CIFAR α = 1: the augmentation parameters of the two views have max |corr| 0.018, and α = 0 is the identity.
- **Probes given ground truth:**
  - content R² = 1.000 and leakage R² = 1.000 from rotated true factors;
  - leakage R² from content alone = −0.0005;
  - shape accuracy = 1.0;
  - orthogonal identifiability is 1.000 for rotated standardised content and 0.74 for a random linear mix;
  - effective rank of N(0, I_12) is 11.995;
  - NP-KSD p-value: 0.71 for N(0,I) vs the Gaussian prior; 0.032 for 1.5·N(0,I) and for N(0,I) vs the uniform
    prior; 0.065 for a uniform sample vs the uniform prior.
- **The "none" arm collapses on real images:**
  - 3DShapes: embedding sd 0.000; content R² 0.07–0.10 (shape accuracy 0.39 vs chance 0.25).
  - CIFAR-10: embedding sd 0.000; class accuracy 0.147 (chance 0.10).
  - The regularised arms keep sd ≈ 1 and content R² 0.4–0.63.

## Pilot (results/pilot_*.parquet, summary in results/pilot_summary.md via tools/pilot_summary.py)
Timing (tools/bench.py, logs/bench/; 3DShapes, d = 12, SIGReg, 256 pairs = 512 images per step):

| setting | ms/step per process | aggregate ms/step |
|---|---|---|
| CPU, 1 thread, ch 64 | 230 | 230 |
| CPU, 1 thread, ch 32 | 146 | 146 |
| CPU, 8 processes × 1 thread, ch 32 | 168 | 21 |
| MPS, 1 process, ch 32 | 18.7–26 | 18.7–26 |
| MPS, 6 processes, ch 64 | 100 | 16.7 |
| **MPS, 6 processes, ch 32 (chosen)** | 43 | **7.2** |
| MPS, 6 processes, ch 64, batch 128 | 43 | 7.2 |

**Device: MPS, 6 concurrent worker processes, 1 CPU thread each.** That is about 3× the throughput of all 8 CPU
cores.

Measured end-to-end rates (wall clock from logs/pilot_round*.log, 6 workers, including evaluation):
- Round 3 (3DShapes, 3000 steps): 12 runs in 6:09 → **30.7 s/run**.
- **Round 5 (2000 steps, mixed tiers): 15 runs in 6:09 → 24.6 s/run.**
- Per-run train time at 2000 steps, with 6 concurrent workers:
  - 3DShapes: 60–93 s for most arms; sliced KSD 177–187 s, the slowest because it is N²×slices;
  - temporal (T=3): 106–164 s;
  - VAE: 77 s.

Pilot outcomes. All rows are single runs at seed 0, so they are not results, and the seed variance is large.
- **3DShapes, 2000 steps, ch 32, round 5:**
  - content R² ranges 0.36 (VICReg) to 0.63 (MMD), with kNN content R² 0.62–0.90;
  - linear leakage ranges 0.00 (MMD) to 0.74 (VICReg), with kNN leakage 0.22–0.73;
  - EMA: content 0.53 at 2000 steps and 0.62 at 3000 steps; linear leakage 0.14 at 2000 steps and 0.05 at 3000.
- **CIFAR-10, 3000 steps:** class accuracy 0.33–0.37 for every regularised arm and EMA. Augmentation leakage is
  low, 0.006–0.05.
- **Temporal (3DShapes clips):** leakage is high for every arm, 0.36–0.83. The target frame's nuisance is one ±1
  step from the context, so it is predictable.
- **VAE tier:** leakage 0.65–0.94, as expected for reconstruction.

### P2 sizing estimate (at 25–30 s/run effective)
- **Tuning**, 3 levels per arm × 2 validation seeds:
  - 3DShapes: 22 configs → 44 runs;
  - temporal: 5 arms → 30 runs;
  - CIFAR: 7 arms → 42 runs;
  - VAE: 18 runs.
- **Held-out**, 5 seeds:
  - H1: 4 d × 8 arms = 32 configs;
  - H2: 2 new σ × 2 d × 5 arms = 20;
  - H3: about 14;
  - H4: 2 new α × 6 arms = 12;
  - temporal 5, CIFAR 8, VAE 3;
  - total about 94 configs → 470 runs.
- **Total**, all tiers: about 600 runs × 27 s ≈ **4.5 h**. P2 has 6 h, which leaves 4.2 h after the 30% margin, so
  this does **not** fit.
- **Core + temporal** only: about 490 runs ≈ **3.7 h**, which fits.
- Recommendation for P2:
  - follow the cut order, VAE tier first; keep CIFAR only if it still fits, for example with fewer tuning levels;
  - or cut steps or seeds per slice. Seeds may not be cut.
  - P2 owns the final sizing.

## Decisions
### Data and views
- **Content and nuisance coding (k = 6).**
  - Content: [cos h_obj, sin h_obj, scale, regular-simplex code of shape (3 dims)].
  - Nuisance: [cos fh, sin fh, cos wh, sin wh, orientation] (5 dims).
  - So H1's d grid is {3, 6, 12, 24}, and the base config is d = 12 (= 2k).
- **Split.** A fixed random 10% of the 480000 factor combinations (seed 12345) is the test split. Training pairs
  never use it.
  - Probes are fit on 10000 train-split images and evaluated on 5000 held-out test-split images.
  - Both sets are drawn from the run's latent law, without view noise.
- **View noise.** Pixel noise ε has sd 0.02 (images in [0, 1]).
- **Latent law.**
  - It acts on object hue (10 values) and scale (8 values); shape stays uniform.
  - The law is discretised over a grid spanning ±2.5 sd. Laplace uses the same unit-variance scale.
  - Hue is treated as an ordered index, so the law is a bump at the middle hue.
- **Loader.** 3DShapes is read with `os.pread` from the memory-mapped .npy file.
  - All workers share one page-cache copy (1.47 GB).
  - A memmap fancy-index gather costs about 75 ms per batch (page faults); pread costs about 1 ms.
- **CIFAR views.**
  - Augmentation: random-resized crop (min area 1−0.8α, log-aspect ±α·log(4/3)), flip with probability 0.5α, and
    brightness/contrast/saturation factors in 1 ± 0.4α.
  - The 8 parameters are the nuisance.
  - Class is probed on un-augmented images. Leakage is fit and evaluated on half/half of augmented test images.
- **Temporal tier: 3DShapes clips, not Moving MNIST.**
  - Reason: the same ground-truth factors and probes as the core tier.
  - Nuisance (orientation and both hues) takes a ±1 step per frame with probability α.
  - Recommend T = 3: 2 context frames, 1 target. The predictor sees the concatenated context embeddings.
  - The regulariser is applied per frame and averaged. Over all 1024 embeddings it cost 900+ s per run.

### Objective and bottlenecks
- **Information bottleneck.** The predictor (or decoder) input is z + σ · std_batch(z).detach() · ξ.
  - The noise is scaled to the detached batch sd, so per-dimension SNR = 1/σ², whatever the scale of z.
  - Without this, the EMA and none arms could escape the noise by growing z.
  - Probes use the clean z.
- **Image JEPA loss (all arms).** Symmetric MSE(pred(bottleneck(z1)), z2) + λ · mean(R(z1), R(z2)), with no
  stop-gradient.
- **EMA arm (7):**
  - target = LayerNorm(EMA encoder(x)), as in V-JEPA;
  - τ = 0.999, stop-gradient, no regulariser;
  - BYOL-style τ = 0.99 without target LayerNorm collapsed (embedding sd 0.000). Pilot rounds 1–3.
- **Shared optimiser.** Adam lr 1e-3 for every arm, with the predictor at 10× lr (1e-2).
  - EMA needed this, and it was neutral to slightly positive for the regularised arms (round 4 vs round 3).
- **Encoder and predictor (all arms):**
  - Conv 3→16→32→32 (4×4, stride 2), then MLP 512→256→d, no normalisation layers, so collapse stays visible;
  - predictor MLP d→256→d;
  - batch 256 pairs;
  - **2000 steps.** Longer is not better: SIGReg and MMD linear content R² fell at 6000 vs 3000 steps, round 2.
- **Width 32, not 64.** That gives 2.3× throughput, with similar content R² for KSD, sliced KSD and MMD; SIGReg
  0.42–0.49 vs 0.57.

### Regularisers
- **SIGReg** is trained with the official `lejepa` SlicingUnivariateTest(EppsPulley(t_max 3, 17 points), 256
  slices).
  - Our re-implementation, `losses.sigreg_ref`, is used only for validation.
  - Removing lejepa's per-call `.item()` sync saved only about 7%, so it was not worth deviating.
- **Sliced KSD.** Gong et al. SKSD with r = g, **32** random unit directions resampled each step. 64 directions
  cost 2× the time for no gain in the pilot.
- **IMQ kernel.** c = 1, β = −1/2 everywhere (Gorham & Mackey).
- **MMD.** Unbiased MMD² against a fresh prior batch, with a sum of Gaussian kernels at bandwidth h·√d,
  h ∈ {0.5, 1, 2}.
- **VICReg.** Variance + 0.04 × covariance, the paper's 25:25:1 expressed relative to invariance = the prediction
  loss.
- **Oracle.** KSD to N(0, diag(1_k, ε² 1_{d−k})) with **ε = 0.3**.
  - ε = 0.1 was too stiff: effective rank 2.3 and content 0.19.
  - It equals N(0, I) when d ≤ k.
- **Matched-prior scores.**
  - Laplace: −tanh(x/0.1)/b, b = 1/√2.
  - Uniform: smooth surrogate, the generalised normal p = 8 with unit variance.
  - MMD and NP-KSD draw from the exact uniform U(−√3, √3).
- **Matched-prior KSD arms** are selected with `prior: laplace|uniform` and `law:` set to the same value.

### Metrics
- **kNN probes added.** kNN R² (k = 10) for content and leakage sits beside the linear probes. Pilots showed
  leakage that is nonlinear only: SIGReg linear leakage 0.08 vs kNN 0.35.
- **NP-KSD metric.**
  - The prior is a product law, so coordinate conditionals equal the marginals.
  - Score: 1-D Gaussian-KDE score from 4000 prior samples.
  - Kernel: Gaussian base kernel, median bandwidth, V-statistic.
  - Null: 30 simulated prior draws.
- **Effective rank** = RankMe. It is meaningless under collapse, so read it with `emb_std_mean`.
- **Tuning criterion (proposed for P2).** Mean validation linear content R² over the 2 validation seeds. EMA tunes
  τ ∈ {0.99, 0.996, 0.999} instead of λ, so every arm gets 3 configs.
- **Pilot λ centres:**
  - VICReg 5, SIGReg 0.05, sliced KSD 5, KSD 1, MMD 10, oracle 0.1;
  - VAE: β-VAE 1, KSD-AE 10, MMD-AE 100 (reconstruction = per-image SSE);
  - suggested 3-level grids: ×{1/5, 1, 5}.
- **No mid-run checkpoints.** Runs take 1–3 min, so resumability is per run via the hash. Changing `DEFAULTS`
  changes every hash, so P2 must not edit defaults mid-sweep.

## Problems
- **Swap thrash.** The first round-2 attempt copied the image array into each of 6 MPS processes, about 9 GB, and
  the machine swapped (9.6 GB swap).
  - The command hit the 45-min foreground limit, and the harness moved it to the background. I stopped it, killed
    the workers, and deleted 7 partial run dirs.
  - Fixed with the pread loader. No results from that attempt are used.
- **Shell bug.** A smoke test passed overrides as one string (zsh does not word-split), so 7 runs trained with a
  malformed config.
  - They were deleted.
  - `resolve()` now rejects unknown keys and bad tier/arm/law values.
- **Laplace KSD has low power.** At n = 256, d = 12, IMQ KSD barely separates a Laplace sample from a Gaussian:
  - full KSD z ≈ 0.2–0.7 across c ∈ {0.5, 1, 2, 4};
  - sliced KSD z ≈ 1.2–2.0;
  - against uniform, full KSD z ≈ 1.3–1.8 and sliced KSD z ≈ 1.2–3.7.
  - This is a real limitation for H3 (the matched-Laplace arm). Report it, don't hide it. Kept c = 1 for
    consistency.
- **Seed variance is large in single pilot runs.** For example, VICReg linear leakage was 0.00, 0.52 and 0.74 across
  the round 2, 4 and 5 settings, and sliced KSD 0.13–0.46. Five held-out seeds with bootstrap CIs are essential,
  and the λ choice may be noisy.
- **Linear content R² is modest**, 0.4–0.63, while kNN reaches 0.75–0.95. 3DShapes content is discrete (320
  combinations), and the regularisers do not force a linear arrangement. This bears on identifiability, so it is a
  finding rather than a bug.
- **NP-KSD p-values sit at the floor.** They are 1/31 for every trained embedding, so all are rejected. Use the
  statistic, not the p-value, to compare runs.
- **Temporal-tier leakage is high for every arm**, because nuisance is predictable from the context frames. H5 is
  still testable, as a relative comparison.
- **Outside the repo:** I created and immediately deleted an empty `/tmp/prof.py` (a typo in one command).
- **MPS kernels may not be bitwise deterministic.** Seeds are fixed, but bitwise reproducibility on MPS is not
  guaranteed. Say so in REPRODUCE.

## Files
- `stein/`:
  - `prep.py` (data → data/processed);
  - `data.py` (samplers);
  - `losses.py` (regularisers, priors);
  - `models.py`;
  - `metrics.py`;
  - `train.py` (entry point);
  - `sweep.py` (multiprocessing grid runner, PROGRESS lines, parquet collection);
  - `validate.py`.
- `tests/` (pytest, 28 tests), `conftest.py`.
- `tools/bench.py`, `tools/show.py`, `tools/pilot_summary.py`.
- `configs/base.yaml`, `configs/pilot/*.yaml`.
- `results/validation.json`, `results/pilot_*.parquet`, `results/pilot_summary.md`, `results/schema.json`.
- Logs: `logs/validate.out`, `logs/pilot_round*.log`, `logs/bench/`.

## Next (P2)
- Write `stein/grids.py`, which returns config lists for `python -m stein.sweep <gridname> --workers 6 --out
  results/<x>.parquet`.
- Handoff 1 is tuning, on validation seeds {100, 101}. Handoff 2 is held-out seeds {0, 1, 2, 3, 4} at the tuned λ.
- Add 95% bootstrap CIs in an analysis script.
- Cut the VAE tier first if the estimate does not fit.
