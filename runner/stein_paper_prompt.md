# Stein regularisation under JEPA bottlenecks, on real data — study prompt

## Role and goal
Produce a reproducible empirical study, written up as a **15-page CS-style paper**
(single column, including references; appendix allowed within the 15 pages), on how
distributional regularisers — Stein discrepancies above all — behave under JEPA
bottlenecks when paired views share signal but carry independent nuisance. The core is
image JEPA; a temporal tier covers V-JEPA-style prediction across frames; an optional
tier asks the same questions of VAE-style autoencoders. This is an analysis study, not a
new-method paper.

**Never fabricate findings, numbers, or citations.** Every number in any output must be
produced by a script in this repo. Report null and negative results as such. If something
cannot be verified, say so explicitly.

## How this run works (headless and unattended)
- You run non-interactively and nobody is watching: no one can answer a question or
  approve a gate until morning. When something is ambiguous, take the most reasonable
  reading and record it under "Decisions" in the current gate file.
- The study is split into phases P0–P4. A runner script starts each phase in a fresh
  session and tells you which phase it is. **Do only that phase.** Earlier phases live on
  disk: read their gate files, not their transcripts.
- Gates are approved automatically, so you are your own reviewer. A phase is complete
  only when its exit criteria hold. If a P1 validation gate fails, fix it in P1; if you
  cannot, record that and drop the arm or tier rather than sweep something that failed
  validation.
- Create `gates/GATE_<n>.md` at the **start** of the phase and keep it current
  (Done / Decisions / Problems / Next). The budget cap can cut a run off between any two
  steps, and the next session resumes from this file.
- End every phase by writing exactly one word to `gates/P<n>.status`, then ending the run:
  - `complete` — the phase's exit criteria hold and the gate file is final;
  - `compute` — you are handing a long job to the runner (below);
  - `blocked` — later phases cannot proceed; say why in the gate file. The runner stops
    the study.
  If you end a run without writing a status, the runner resumes you and asks again.
- **Long jobs go to the runner.** A foreground command may run for up to 45 minutes (Bash
  timeout up to 2700000 ms), and waiting on one costs nothing, so run tests and pilots in
  the foreground. Never start a background job and poll it: polling costs tokens, and the
  job dies with the run. For anything longer than about 40 minutes, write `jobs/P<n>.sh`
  and hand it off:
  - it runs from the repo root as `bash jobs/P<n>.sh`, uses `.venv/bin/python`, and is
    idempotent — rerunning it skips finished work, so a job that is cut off loses nothing;
  - it prints a line starting with `PROGRESS` at least every 10 minutes (for example
    `PROGRESS 120/480 runs, ~2.1 h left`), which the runner copies into the progress log;
  - test it on a tiny slice yourself first;
  - then write `compute` to the status file, update the gate file, and end the run.

  The runner executes the job within the phase's compute allowance (see "Run context"),
  then resumes this session with the exit code and the tail of
  `logs/compute-P<n>-<k>.log`. It accepts at most two handoffs per phase. A job stopped by
  the allowance is not a failure: analyse what finished, and size anything further from
  the measured rate.
- Be economical with tokens. Don't re-read large files, don't fetch whole PDFs when an
  abstract or a section is enough, and write large outputs to files instead of printing
  them.
- Python is `.venv/bin/python` only (Python 3.12; the runner preinstalled torch, numpy,
  scipy, pandas, pyarrow, scikit-learn, h5py, pyyaml, pytest, joblib, matplotlib, and
  `lejepa` from GitHub). Add packages with `uv pip install --python .venv/bin/python <pkg>`,
  and keep `requirements.txt` pinned to what is actually installed
  (`uv pip freeze --python .venv/bin/python`).
- Work only inside this directory, and leave `runner/` alone: it is the runner itself.
  Don't install system packages or touch files elsewhere on the machine, and don't run git
  commands that write (commit, checkout, reset, stash): the runner commits after every
  phase and before every handoff.

**Budget.** The runner sets each phase's dollar budget and prints it under "Run context";
unspent money rolls forward to the next phase. The plan is P0 ≈ $12 · P1 ≈ $24 · P2 ≈ $18 ·
P3 ≈ $16 · P4 ≈ $10, with about $10 held back for wrap-ups. Scope each phase to fit.
If scope must shrink, cut in this order: VAE tier → natural-image tier → full MMD arm →
uniform latent law → third λ level → temporal tier. **Never cut** seeds, validation gates,
or the independent review.

## Compute and data
All compute is local, on the machine described under "Run context": CPU, or Apple's GPU
through PyTorch MPS if the pilot shows it is faster. No cloud compute.

The study uses real image datasets with known generative factors, so every metric has a
ground truth. The runner has already downloaded them to `data/raw/`:
- **3DShapes** (Burgess & Kim, 2018) — `data/raw/3dshapes.h5`: 480,000 rendered RGB 64×64
  scenes covering a full grid of six factors (floor hue, wall hue, object hue, scale,
  shape, orientation). The core dataset.
- **CIFAR-10** — `data/raw/cifar-10-python.tar.gz`: natural images, for the natural-image
  tier.
- **MNIST** — `data/raw/mnist/`: for Moving-MNIST clips, if the temporal tier uses them.

Download no other datasets. Preprocess each once into memory-mapped uint8 `.npy` files
under `data/processed/` and have workers memory-map them: 3DShapes at full 64×64 is
5.9 GB and cannot be held by every worker, while 32×32 is 1.5 GB. Keep total RAM use under
about 12 GB. Synthetic samples are for unit tests and validation gates only; no reported
result comes from synthetic data.

---

## P0 — Related work and novelty
For each paper below, verify its existence and details against arXiv/DBLP/the venue, then
write `notes/<citekey>.md` with: claim · method · overlap with this study · open gap ·
verified BibTeX (fetched, not typed from memory). Read the abstract and the relevant
sections, not whole PDFs.

**Use the `lit-reviewer` sub-agent, one paper per call**, at most four calls at a time
(the run bills an API key, whose rate limits a wider fan-out would hit). Do the synthesis
yourself.

Core:
- LeJEPA / SIGReg — Balestriero & LeCun, arXiv:2511.08544 (code: github.com/rbalestr-lab/lejepa)
- KerJEPA — Zimmermann, Wiltzer, Szeto, Alvarez-Melis & Mackey, arXiv:2512.19605.
  Already uses KSD/MMD (sliced and analytic, Gaussian/IMQ kernels) as JEPA regularisers.
  This is the main overlap.
- "When Does LeJEPA Learn a World Model?" — Klindt, LeCun & Balestriero, arXiv:2605.26379
  (linear identifiability only under Gaussian latents)
- VICReg — Bardes, Ponce & LeCun, ICLR 2022
- Sliced Kernelized Stein Discrepancy — Gong, Li & Hernández-Lobato, ICLR 2021
- V-JEPA — Bardes et al. 2024; V-JEPA 2 — Assran et al. 2025 (collapse prevented by
  EMA + stop-gradient, no distributional regulariser)
- I-JEPA — Assran et al. 2023
- von Kügelgen et al. 2021 — content/style identifiability in SSL

Stein background:
- Liu, Lee & Jordan 2016 (KSD); Chwialkowski, Strathmann & Gretton 2016
- Gorham & Mackey 2017 (IMQ kernel, measuring sample quality)
- Anastasiou et al., "Stein's Method Meets Computational Statistics" (arXiv:2105.03481)
- Reinert & Röllin 2009, multivariate exchangeable pairs and singular normal targets (arXiv:0711.1082)
- Xu & Reinert, NeurIPS 2022, NP-KSD for implicit generative models (arXiv:2206.00149)
- Mijoule, Reinert & Swan, Stein's density method for multivariate continuous distributions
- Xu, Generalised KSD (arXiv:2106.12105); Ley, Reinert & Swan 2017 (univariate Stein operators)

Autoencoders (for the VAE tier):
- Kingma & Welling 2014, Auto-Encoding Variational Bayes; Higgins et al. 2017, β-VAE
- Tolstikhin, Bousquet, Gelly & Schölkopf 2018, Wasserstein Auto-Encoders (MMD on the
  aggregate posterior)
- Zhao, Song & Ermon, InfoVAE
- Pu et al. 2017, VAE learning via Stein variational gradient descent

Data and evaluation:
- Burgess & Kim 2018, the 3D Shapes dataset (github.com/google-deepmind/3d-shapes)
- Locatello et al. 2019, "Challenging Common Assumptions in the Unsupervised Learning of
  Disentangled Representations" (factor datasets, and what can be identified without
  supervision)
- Krizhevsky 2009 (CIFAR-10); Srivastava, Mansimov & Salakhutdinov 2015 (Moving MNIST)

Context:
- Expanding SPHERE-JEPA (arXiv:2606.17603); VISReg (arXiv:2606.02572)
- RankMe (Garrido et al. 2023); Wang & Isola 2020; Federici et al. 2020 (multi-view IB)

Then write `notes/related_work.md` with an **overlap table**: rows are papers; columns are
regulariser · sliced/full · prior · nuisance · dimensional bottleneck · information
bottleneck · identifiability · objective (JEPA / temporal JEPA / autoencoder) · data
(synthetic / rendered / natural). Claim novelty **only** for cells the table shows
uncovered. Write `paper/refs.bib` from the verified entries only. Mark anything
unverifiable.

→ Write `gates/GATE_0.md`, write `complete` to `gates/P0.status`, and end the run.

---

## P1 — Implementation and validation
This directory starts with only the runner's files. Lay the code out under `stein/`:
plain PyTorch, YAML configs, one entry point per experiment, runs keyed by config hash
and resumable, fixed seeds, `requirements.txt` pinned.

**Views.** Paired views share signal and carry independent nuisance, with ground truth
for both.
- *3DShapes (core).* Shared content s = object shape, scale, and object hue; nuisance
  n = floor hue, wall hue, and orientation. For each pair, draw s once; for the second
  view, resample each nuisance factor independently with probability α (α = 0: the views
  share their nuisance too; α = 1: nuisance is independent across views). Small pixel noise
  is the view noise ε. Encode hues, which are circular, as (cos, sin) for the probes, and
  state k, the dimension of s as the probes see it, in the gate file.
- *Natural-image tier (CIFAR-10).* Two independent augmentations of one image. The
  augmentation parameters drawn for each view (crop box, scale, flip, colour jitter) are
  the nuisance: known, and independent across views. α scales augmentation strength. The
  shared-signal probe is the class label.
- *Temporal tier (V-JEPA-style).* Short clips in which content is fixed and nuisance moves
  frame to frame: either 3DShapes with orientation (and optionally the hues) walking over
  time, or Moving MNIST generated from MNIST, which keeps the digit labels. The objective
  predicts target-frame embeddings from context frames, against an EMA target encoder
  with stop-gradient, as V-JEPA does. Choose one and record why.
- *VAE tier (optional).* The same encoder backbone with a decoder, on 3DShapes only.

**Axes**
- embedding dim d (dimensional bottleneck)
- Gaussian noise of variance σ² injected into z (information bottleneck: a capacity
  limit, distinct from d)
- nuisance strength α, as defined per dataset above
- shared-latent law ∈ {uniform, Gaussian, Laplace}: reweight how often each content-factor
  value is sampled (uniform is the dataset's own grid; the others are discretised). The
  images stay real; only their sampling law changes.
- regularisation weight λ

**Encoder.** One small convolutional encoder and one MLP predictor for every arm, sized
from the pilot (≤ 64 channels at 32×32 is a reasonable start), with the same optimiser
and step count.

**Arms** (same encoder, predictor, and optimiser for all):
1. none (collapse check)
2. VICReg
3. SIGReg (the official `lejepa` package as the reference implementation)
4. sliced KSD (IMQ kernel)
5. full KSD (IMQ kernel)
6. MMD (Gaussian kernel)
7. EMA target + stop-gradient (V-JEPA/BYOL-style, no distributional regulariser)
8. oracle diagnostic: KSD toward a rank-k Gaussian (k known). Report it as a diagnostic,
   not a practical method.

Temporal tier: EMA alone (the V-JEPA recipe), EMA + sliced KSD, EMA + SIGReg, sliced KSD
alone, VICReg. VAE tier: β-VAE (KL to N(0, I)); an autoencoder whose aggregate posterior is
pushed toward the prior by KSD with an IMQ kernel (Stein needs only the prior's score, so
it can target a non-Gaussian prior); and the same with MMD, as in WAE-MMD. Where a matched
prior's score is undefined (uniform), use a smooth surrogate and say which.

**Metrics**
- probe R² for s (linear probe on frozen embeddings of held-out images; also accuracy for
  categorical factors)
- nuisance leakage R² for n (lower is better)
- linear identifiability (R² after an orthogonal fit to the standardised content factors)
- effective rank
- Gaussianity statistic, and NP-KSD fit to the intended prior
- loss stability
- natural-image tier: linear-probe class accuracy, and leakage R² of the augmentation
  parameters

**Validation gates.** All must pass before any sweep; write the results to `gates/GATE_1.md`:
- each regulariser ≈ 0 on N(0, I) samples and clearly larger on collapsed or anisotropic
  samples
- gradient checks pass; one unit test per loss (pytest)
- SIGReg matches the `lejepa` package on the same batch
- the pair sampler does what it claims: content identical across views; at α = 1,
  nuisance uncorrelated across views
- probes recover ground-truth factors when given them
- the "none" arm visibly collapses on real images

Run a small **pilot**: one short run per arm, per tier you intend to keep, timed on CPU and
on MPS. Record the runtime per run, the device you chose and why, and an estimate for the
P2 grid at that rate.

→ Write `gates/GATE_1.md`, write `complete` to `gates/P1.status`, and end the run.

---

## P2 — Sweep and analysis
Design the grid as hypothesis-driven slices around one base configuration (for example
3DShapes, d = 2k, α = 1, σ = 0, uniform law, tuned λ), not as a full factorial:
- H1: d spanning below and above k (for example {k/2, k, 2k, 4k}) × the arms
- H2: σ ∈ {0, mid, high} × two values of d
- H3: latent law × {KSD with the matched prior, the Gaussian-target regularisers, oracle}
- H4: α at 3 levels × {EMA, the distributional regularisers}
- each further tier: one slice, only if it fits

Size the sweep from the pilot so it fits the phase's compute allowance with a 30% margin,
and hand it to the runner. Two handoffs fit naturally: tuning first, then held-out seeds.
Shrink first if the pilot says it won't fit; expand only if there is budget.

- Parallelise in code (multiprocessing over the grid, one thread per worker). **Never**
  spawn sub-agents per run.
- **Matched tuning:** the same number of configs per arm, tuned on 2 validation seeds;
  report ≥5 held-out seeds with 95% bootstrap CIs.
- All results go to `results/*.parquet`, with the schema documented in `results/schema.json`.

**Hypotheses.** Give each a verdict: confirmed / refuted / inconclusive, with the evidence.
- **H1:** for d > k, isotropic-Gaussian regularisers increase nuisance leakage compared
  with VICReg and EMA.
- **H2:** the information bottleneck (σ) reduces leakage where the dimensional one (d)
  does not.
- **H3:** with non-Gaussian content laws, KSD with a matched prior beats Gaussian-target
  regularisers on identifiability.
- **H4:** EMA/stop-gradient behaves qualitatively differently from distributional
  regularisers under nuisance.
- **H5 (temporal tier):** with nuisance that moves over time, EMA + stop-gradient alone
  leaks less nuisance than any distributional regulariser alone, and adding sliced KSD to
  EMA leaves leakage unchanged within the CIs.
- **H6 (VAE tier):** at matched d, the reconstruction-based models leak more nuisance than
  every JEPA arm, because reconstruction requires it. Also report whether H3's
  matched-prior effect carries over to the KSD autoencoder.

A tier that was cut gets the verdict "not tested", with the reason.

→ Write `gates/GATE_2.md` (key tables, verdicts, surprises), write `complete` to
`gates/P2.status`, and end the run.

---

## P3 — Outputs
Build one Quarto project in two formats. Every number and figure comes from `results/`
through the same scripts; no hand-typed numbers.

**Paper** (PDF, ≤ 15 pages including references). Render it with Quarto's `typst`
format, which needs no LaTeX; a LaTeX engine is fine instead if one works on this machine.
- abstract
- introduction with an explicit list of contributions
- related work, built on the overlap table
- setup: datasets and view construction, bottleneck definitions, arms, metrics, tuning
  protocol
- results, organised by hypothesis
- discussion: dimensional vs information bottleneck
- limitations: the core dataset is rendered, the natural-image nuisance is defined by
  augmentation, small scale, the oracle arm, a laptop's compute budget
- future work: Causal3DIdent, MPI3D-real, ImageNette, real video for the temporal tier
- reproducibility statement

Vector figures (PDF/SVG), 5–7 main figures, verified BibTeX only.

**Blog post** (HTML) in `blog/`: a plain-language version with lightweight Observable JS
controls (d, α, σ, arm) over the precomputed results. No server. Write it to
`ref/STYLE.md`, the blog's style guide (Register B for a technical post). It will later be
moved into that blog, so keep it self-contained: it reads precomputed files under
`blog/data/` and executes no Python when it renders.

**README:** exact commands, seeds, environment, hardware, runtime.

If useful, use a general-purpose sub-agent to build the Quarto scaffolding against
`results/schema.json` while you write the text, but integrate and check it yourself.

→ Write `gates/GATE_3.md` (page count, figure list, render commands), write `complete` to
`gates/P3.status`, and end the run.

---

## P4 — Independent review
Use a fresh general-purpose sub-agent that has not seen the work. Give it only the repo
path and this checklist:
- every number in the paper and the blog traces to `results/`
- novelty claims are consistent with `notes/related_work.md`
- hypothesis verdicts follow from the CIs, and every tier claimed was actually run
- every BibTeX entry resolves to a real paper with correct authors, year, and venue
- the paper renders and is ≤ 15 pages
- the README commands reproduce one figure from scratch

Fix the issues it finds, then write `gates/GATE_4.md` and `FINAL.md` (summary, issues
fixed, known gaps), write `complete` to `gates/P4.status`, and end the run.
