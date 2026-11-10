# Why distributional pressure buys nuisance: a linear theory of JEPA regularisers and its test — study 2 prompt

## Role and goal
This workspace holds a finished study ("study 1": `FINAL.md`, `paper/`, `results/`,
`gates/`). You are running its follow-up. Study 1 was a broad empirical comparison; it
refuted five of its six hypotheses and left three findings unexplained:

1. **Leakage follows λ, not the target law.** For every regulariser, VICReg included,
   nuisance leakage rose with the regularisation weight, often abruptly (KSD at d = 12:
   linear leakage 0.003 at λ = 0.2, 0.578 at λ = 1). Which arm "leaks least" was decided by
   the tuning rule, so point comparisons at one tuned λ were fragile.
2. **Noise on the predictor input (σ) raised leakage** instead of lowering it, while
   narrowing d lowered it.
3. **A matched non-Gaussian prior did not improve identifiability**, measured by R² after
   an orthogonal fit.

Study 2 is narrower and deeper: **derive a small theory that explains these findings,
state its predictions before running anything, then test them** on the same data with
the same code. The output is a **paper of at most 12 pages** (single column, references
and appendix included) with a theory section, proofs in the appendix, and experiments
organised around the predictions. It is an explanation-and-test paper, not a new-method
paper.

**Never fabricate findings, numbers, citations, or proofs.** Every number comes from a
script in this repo. A statement is called a theorem, proposition or lemma only if the
paper contains a complete proof that you have checked line by line and confirmed
numerically; anything else is labelled a conjecture or a heuristic. If a statement below
turns out false, say so and report the corrected statement: a refuted conjecture is a
result. Report null and negative results as such.

## How this run works (headless and unattended)
- You run non-interactively and nobody is watching: no one can answer a question or
  approve a gate until morning. When something is ambiguous, take the most reasonable
  reading and record it under "Decisions" in the current gate file.
- The study is split into phases P0–P4. A runner script starts each phase in a fresh
  session and tells you which phase it is. **Do only that phase.** Earlier phases live on
  disk: read their gate files, not their transcripts.
- **Study 2's control files live under `study2/`:** `study2/gates/GATE_<n>.md`,
  `study2/gates/P<n>.status`, `study2/jobs/P<n>.sh`, `study2/logs/`. Where the workspace's
  `AGENTS.md`/`CLAUDE.md` says `gates/` or `jobs/`, read `study2/gates/` and `study2/jobs/`.
  The top-level `gates/`, `jobs/`, `logs/` and `state/` belong to study 1: never write
  there.
- Gates are approved automatically, so you are your own reviewer. A phase is complete
  only when its exit criteria hold.
- Create `study2/gates/GATE_<n>.md` at the **start** of the phase and keep it current
  (Done / Decisions / Problems / Next). The budget cap can cut a run off between any two
  steps, and the next session resumes from this file.
- End every phase by writing exactly one word to `study2/gates/P<n>.status`, then ending
  the run:
  - `complete` — the phase's exit criteria hold and the gate file is final;
  - `compute` — you are handing a long job to the runner (below);
  - `blocked` — later phases cannot proceed; say why in the gate file. The runner stops
    the study.
  If you end a run without writing a status, the runner resumes you and asks again.
- **Long jobs go to the runner.** A foreground command may run for up to 45 minutes (Bash
  timeout up to 2700000 ms), and waiting on one costs nothing, so run tests and pilots in
  the foreground. Never start a background job and poll it. For anything longer than about
  40 minutes, write `study2/jobs/P<n>.sh` and hand it off:
  - it runs from the repo root as `bash study2/jobs/P<n>.sh`, uses `.venv/bin/python`, and
    is idempotent — rerunning it skips finished work;
  - it prints a line starting with `PROGRESS` at least every 10 minutes;
  - test it on a tiny slice yourself first;
  - then write `compute` to the status file, update the gate file, and end the run.

  The runner executes the job within the phase's compute allowance (see "Run context"),
  then resumes this session with the exit code and the tail of the job's log. It accepts
  at most two handoffs per phase. A job stopped by the allowance is not a failure: analyse
  what finished.
- Be economical with tokens: the whole study is capped at $60, two thirds of study 1's cap.
  Don't re-read large files (`results/analysis.md` is long: grep it), don't fetch whole
  PDFs, and write large outputs to files instead of printing them.
- Python is `.venv/bin/python` only. Add packages with
  `uv pip install --python .venv/bin/python <pkg>` (sympy is a reasonable addition for
  checking algebra) and keep `requirements.txt` pinned to what is installed.
- Work only inside this directory. Leave `runner/` and the top-level `README.md` alone.
  Don't run git commands that write: the runner commits after every phase and before every
  handoff.

**Study 1 is a frozen record.** Do not modify `paper/`, `blog/`, `explainer/`,
`results/`, `gates/`, `FINAL.md`, `REPRODUCE.md`, `PROMPT.md` or `notes/related_work.md`.
Read them freely. Study 2 writes to:

| Path | Holds |
|---|---|
| `study2/` | gates, jobs, logs (control files) |
| `theory/` | `theory.md` (statements and proofs), `predictions.md`, `check_*.py` (numerical checks) |
| `stein/`, `tests/`, `configs/` | code, extended in place (rules below) |
| `results2/` | every study-2 result (`*.parquet`, `*.json`, `schema.json`, `analysis.md`) |
| `paper2/` | `paper.qmd`, `_generated/`, `figures/`, `refs.bib`, `paper.pdf` |
| `notes/` | new `notes/<citekey>.md` files only |
| `REPRODUCE2.md`, `FINAL2.md` | reproduction guide and final summary |

Don't name a new tracked directory `data`, `logs` or `state`: `.gitignore` ignores those
names at any depth.

**Extending the code without breaking the cache.** `runs/` holds 844 finished runs keyed
by `cfg_hash` (see `stein/train.py`), and study 2 reuses them, for example the
λ-sensitivity slice. Adding a key to `DEFAULTS` would change every hash. So: any new
config key is left **out of the hash while it has its default value**, and its default
reproduces study 1's behaviour exactly. Add a test that recomputes the hash of the config
stored in every `runs/*/result.json` and finds it unchanged, and keep the existing 28
tests passing. Cached results lack study 2's new metrics; where an old run needs them,
give it a new hash through such a key (for example an evaluation-version key) and rerun it,
rather than editing files under `runs/`.

**Budget.** The runner prints each phase's dollar budget under "Run context"; unspent
money rolls forward. The plan is P0 ≈ $13 · P1 ≈ $12 · P2 ≈ $9 · P3 ≈ $12 · P4 ≈ $8, with a
little held back for wrap-ups. If scope must shrink, cut in this order: E6 (theory-guided
arm) → E5 (identifiability) → E4 (noise bottleneck) → the second value of d in E1 → the
smallest α in E1. **Never cut** the pre-registered predictions, the numerical checks of
the theory, the 5 held-out seeds, or the independent review.

## Compute and data
All compute is local, on the machine under "Run context". Study 1 measured about 26 s per
run effective with 6 MPS workers (`gates/GATE_2.md` has per-arm times), so the 6-hour P2
allowance holds roughly 550 runs with a 30% margin. Data is already in `data/`; download
nothing.

Two kinds of evidence, never mixed:
- **Theory checks** use a synthetic linear-Gaussian model. They verify algebra, and are
  reported as such, in the theory section and the appendix.
- **Experiments** use real images (3DShapes, as in study 1), and every empirical claim
  about networks comes from them.

---

## P0 — Theory, predictions, and the literature it needs

### The model
Content s ∈ ℝᵏ and nuisance n ∈ ℝᵐ are independent, zero-mean, with identity covariance.
A pair of views is x = (s, n) and x′ = (s, n′), where each nuisance coordinate is
resampled independently with probability α, so Cov(n, n′) = (1 − α) I. The encoder is
linear, z = A s + B n ∈ ℝᵈ, and z′ = A s + B n′. Study 1's sampler has exactly this
structure, with k = 6.

### Statements to prove, correct, or refute
The sketches below come from the study's design and are **unchecked**. Treat each as a
conjecture: prove it, find the corrected statement, or refute it. State every assumption.

- **T1 (leakage threshold).** With an identity predictor and a covariance penalty,
  J(A, B) = E‖z − z′‖² + λ‖Σ_z − I‖²_F, where Σ_z = AAᵀ + BBᵀ. The invariance term is
  2α‖B‖²_F and content is free. Sketch: if d ≤ k, every global minimiser has B = 0. If
  d > k (and m ≥ d − k), every global minimiser puts unit content variance on a
  k-dimensional subspace and nuisance variance max(0, 1 − α/λ) on each of the remaining
  d − k directions. So nuisance enters only above a threshold **λ\* = α**, and then grows
  as 1 − α/λ.
  Also solve the case **m < d − k**, which is the one the experiments are in: study 1's
  nuisance code has 5 dimensions (`stein/data.py`), so d − k = 6 at d = 12 and 18 at
  d = 24 exceed it. State what the minimisers do with the directions that neither content
  nor nuisance can fill, and carry that case into every prediction.
  Corollaries to establish: (a) a linear probe's leakage R² is scale-invariant, so it is a
  **step** at λ\*, while the share of embedding variance explained by nuisance is
  continuous; (b) λ\* is proportional to α, and at α = 0 nuisance costs nothing at any
  λ > 0; (c) with d ≤ k there is no threshold; (d) the result needs linearity: if the
  encoder may also compute d − k further functions of s that are uncorrelated with the
  first k, J reaches its minimum with no nuisance at all. So in a nonlinear network,
  leakage above the threshold is not forced by the objective. It is a statement about
  which zero-cost solution training finds.
- **T2 (which regularisers, which predictor).** Extend T1 as far as it goes: (i) a learned
  linear predictor W, with loss E‖W z − z′‖² (and with stop-gradient on z′ if tractable);
  (ii) the regularisers actually used. For Gaussian z each of KSD (IMQ), MMD (Gaussian
  kernel) and the Epps–Pulley statistic behind SIGReg is, in population, a function of
  Σ_z minimised at I; expand each to second order around I to get an effective curvature,
  hence a per-regulariser threshold λ\*_reg = c_reg · α (compute c_reg, in closed form or
  numerically). VICReg's variance term is a hinge, not a quadratic: work out what replaces
  the threshold. Say plainly where the extension stops being a proof and becomes a
  heuristic.
- **T3 (noise bottleneck).** Study 1's σ adds noise σ · sd_batch(z) · ξ to the predictor's
  input only. In the linear model with a linear predictor, derive how σ moves the optimum
  and whether the model predicts the observed rise in leakage. Do the same for noise of
  fixed scale (σ in absolute units, with the regulariser holding the scale of z near 1).
  If the linear model predicts no effect, or the wrong sign, report that.
- **T4 (what a matched prior can identify).** (i) Under any regulariser that depends on z
  only through an orthogonally invariant functional (all Gaussian-target arms in
  population), the set of minimisers is closed under rotations of z, so R² after an
  orthogonal fit is the strongest identifiability such an arm can deliver, and a matched
  prior cannot beat it **on that metric**. (ii) If s has independent components, at most
  one of them Gaussian, and z = M s matches a product prior with those marginals exactly,
  then M is a signed permutation (the ICA theorem, via Darmois–Skitovich). So the
  matched-prior effect, if it exists, shows only in a permutation-level metric such as the
  mean correlation coefficient (MCC), which study 1 did not measure. State what this
  requires of the content code (study 1 codes hues as (cos, sin), which are dependent).

For each statement write, in `theory/theory.md`: the precise statement, the proof (or the
counterexample), and its status (proved / corrected and proved / heuristic / refuted).
For each write `theory/check_<name>.py`: gradient descent on the linear model in torch,
from several random starts, compared with the closed form (nuisance variance per
direction against λ, the threshold against α, d ≤ k against d > k), with results written
to `results2/theory_<name>.json` and a pass/fail tolerance. A proof and its check must
agree before the statement is called proved.

### Predictions, fixed before any network is trained
Write `theory/predictions.md`: numbered predictions for experiments E1–E5 (below), each
with the quantity, the direction or functional form, and what outcome would count against
it. Include the ones the theory does **not** make (for example, it says nothing about
which zero-cost solution a nonlinear network picks). Later phases may not edit this file;
the paper reports every prediction's outcome.

### Literature
Study 1's 35 notes stand. Add only what the theory needs, at most 8 papers, with the
`lit-reviewer` sub-agent, one paper per call, at most four at a time. Candidates (verify
each; drop any that cannot be verified, and replace none from memory): Tian, Chen &
Ganguli 2021 (dynamics of non-contrastive SSL in linear models); Jing et al. 2022
(dimensional collapse); Simon et al. 2023 (stepwise SSL learning); Balestriero & LeCun
2022 (SSL methods as spectral embeddings); Shwartz-Ziv et al. (information-theoretic view
of VICReg); Comon 1994 (ICA identifiability); Zimmermann et al. 2021 (contrastive
learning inverts the data-generating process); Shah et al. 2020 (simplicity bias). Then
write `notes/related_theory.md`: one paragraph per paper on whether it already contains
T1–T4 or a close relative. **If a statement is already known, cite it and say so**; the
paper then claims only the application and the test.

→ Exit criteria: every statement has a status and a passing check (or is marked heuristic
or refuted); `theory/predictions.md` exists. Write `study2/gates/GATE_0.md`, write
`complete` to `study2/gates/P0.status`, and end the run.

---

## P1 — Implementation and validation
Extend `stein/` under the cache rules above. Needed:

- **Arm `cov`:** the pure covariance penalty of T1, as the bridge between theory and the
  kernel regularisers.
- **Encoder option `linear`:** a single linear map from pixels, the rung between the
  linear model and the convolutional encoder.
- **Metrics:** (i) nuisance and content **variance share**: the fraction of total
  embedding variance linearly explained by n and by s (regress z on the factor codes),
  which, unlike probe R², sees how much of the embedding the nuisance occupies; (ii)
  **MCC** between embedding coordinates and content factors after the best permutation
  (Hungarian matching on absolute correlations); (iii) per-direction nuisance variance in
  the embedding's principal axes, for comparison with 1 − α/λ.
- **Noise mode:** `fixed`-scale σ beside study 1's batch-scaled σ.
- **Grids** for E1–E6 in `stein/grids.py` (new functions; leave the old ones unchanged)
  and an analysis script writing `results2/`.

**Validation gates** (all must pass; results in `study2/gates/GATE_1.md`):
- the existing tests pass and the cache-hash test passes;
- one unit test per new metric, on synthetic embeddings with a known answer (MCC = 1 for
  a signed permutation and < 1 for a rotation; variance share matching a constructed
  mixture);
- the `cov` arm is ≈ 0 on N(0, I) samples, larger on collapsed or anisotropic ones, and
  passes a gradient check;
- **theory meets code:** the real training loop, run with the `linear` encoder and the
  `cov` arm on synthetic linear-Gaussian views, reproduces T1's threshold (or the
  corrected statement) within the tolerance of `theory/check_*.py`;
- a rerun of one cached study-1 config reproduces its stored metrics.

**Pilot** (foreground): time one run per new arm and encoder; locate the threshold
roughly for each regulariser at α = 1, d = 12 with a coarse λ scan on one seed (the cached
λ-sensitivity runs already give three levels per arm), so that P2's λ grid brackets it.
Pilot runs use seed 0 and are never reported.

→ Write `study2/gates/GATE_1.md` (gates, timings, the λ range per arm, the P2 run-count
estimate), write `complete` to `study2/gates/P1.status`, and end the run.

---

## P2 — Experiments and analysis
Base setting as in study 1: 3DShapes, d = 12, α = 1, σ = 0, uniform law, 2000 steps,
held-out seeds 1–5 (reuse cached runs wherever the config matches). **There is no λ
tuning in study 2:** arms are compared along their whole λ path, which removes the
tuning-rule dependence that undid study 1's H1. Size everything from the P1 pilot to fit
the compute allowance with a 30% margin; one handoff should suffice, a second is
available.

- **E1 — λ paths and frontiers (the core).** Arms {cov, VICReg, SIGReg, sliced KSD, KSD,
  MMD} × about 7 log-spaced λ levels bracketing each arm's threshold × α ∈ {0.25, 0.5, 1}
  at d = 12, and α = 1 at d ∈ {6, 24}. Report, per arm: leakage (linear R², kNN R²,
  variance share) and content R² against λ; the estimated threshold λ̂\* with a bootstrap
  CI (define the estimator in the gate file before looking at held-out runs, for example
  the λ at which leakage first exceeds a fixed level, interpolated in log λ); and the
  **content–leakage frontier**. Compare arms by leakage at matched content R² and by the
  best content R² reachable under a leakage ceiling, not at a single λ.
- **E2 — the theory's quantitative tests**, read off E1 plus the `linear`-encoder rung
  (cov and KSD, same λ grid, α ∈ {0.5, 1}): does λ̂\* scale with α; is there a threshold at
  d = 6 ≤ k; does per-direction nuisance variance follow 1 − λ\*/λ above the threshold (and the
  variance share the form that implies) while probe R² steps;
  do the per-regulariser thresholds order as c_reg predicts?
- **E3 — why nuisance, when content would be free (T1d).** At two λ levels above the
  threshold, for KSD and SIGReg: training length 2000 against 8000 steps. Does leakage
  fall with longer training (a transient of easy features) or persist? Also report what
  fills the spare dimensions: variance share of s, of n, and unexplained.
- **E4 — noise bottleneck (T3).** Batch-scaled against fixed-scale σ ∈ {0, 0.5, 1} for KSD
  and VICReg at d = 12, against T3's prediction.
- **E5 — identifiability with the right metric (T4).** Study 1's H3 cells at the Laplace
  and uniform laws (Gaussian-target KSD, matched-prior KSD, SIGReg, MMD), re-run with MCC
  beside the orthogonal-fit R², at the λ levels study 1 selected and one level below.
  Report also the cell where T4's assumptions hold best, if P0 identified one.
- **E6 — optional, first to be cut.** If the theory suggests a concrete remedy (for
  example a λ schedule held below the threshold, or a penalty that leaves spare
  directions unconstrained), test one such arm on the E1 frontier. Report it as a check
  of the theory, not as a method.

All results go to `results2/*.parquet` with `results2/schema.json`; 95% bootstrap CIs
over the 5 seeds; contrasts within about 0.01 of zero are read as weak, as in study 1.

**Verdicts.** For every numbered prediction in `theory/predictions.md`: confirmed /
refuted / inconclusive, with the evidence. Then answer in the gate file, plainly: which of
study 1's three unexplained findings does the theory now explain, which not, and does the
ranking of arms change once they are compared by frontier?

→ Write `study2/gates/GATE_2.md` (key tables, verdicts, surprises), write `complete` to
`study2/gates/P2.status`, and end the run.

---

## P3 — Paper
One Quarto document, `paper2/paper.qmd`, rendered with typst to `paper2/paper.pdf`
(`quarto render paper2/paper.qmd`; add it to `_quarto.yml` only if study 1's two
documents still render unchanged). At most 12 pages. As in study 1, the `.qmd` executes
no Python: a report script writes `paper2/_generated/` and `paper2/figures/` from
`results2/`, and no number is typed by hand.

- abstract; introduction with the three findings of study 1 that motivate it (cite
  `paper/paper.pdf` as the companion study) and an explicit list of contributions
- the model and the results T1–T4, each with its status; proofs in the appendix,
  complete enough to check
- predictions, as registered, and the experiments E1–E6 organised by prediction, with
  the theory's curve drawn on the data wherever it makes one
- what the theory does not explain
- related theory, built on `notes/related_theory.md`, with honest novelty claims
- limitations (a linear model of a nonlinear network; one rendered dataset; small
  models; 5 seeds), reproducibility statement (an AI agent ran the experiments and
  drafted the text and proofs; the proofs were machine-checked numerically, not formally)
- 4–6 vector figures, verified BibTeX only (`paper2/refs.bib`, built from the notes)

Also write `REPRODUCE2.md`: exact commands for the theory checks, one experiment figure,
and the paper; seeds, environment, runtime. No blog post in this study.

→ Write `study2/gates/GATE_3.md` (page count, figure list, render commands), write
`complete` to `study2/gates/P3.status`, and end the run.

---

## P4 — Independent review
Use two fresh general-purpose sub-agents that have not seen the work, each given only the
repo path and its checklist.

**Proof referee:**
- every statement called a theorem, proposition or lemma has a complete proof; check
  each step, and try to break each statement with a small counterexample (it may write
  and run its own numerical checks, independent of `theory/check_*.py`)
- the assumptions stated are the ones the proof uses
- the paper's wording never claims more than the status in `theory/theory.md`
- claims of novelty are consistent with `notes/related_theory.md`

**Empirical reviewer:**
- every number in the paper traces to `results2/` (or `results/` for study-1 numbers)
- `theory/predictions.md` was not edited after P0 (compare with the P0 commit in
  `git log`), and every prediction's outcome is reported, including the failed ones
- verdicts follow from the CIs
- every new BibTeX entry resolves to a real paper with correct authors, year, and venue
- the paper renders and is ≤ 12 pages; study 1's files are unmodified (`git diff` against
  the commit before study 2 for `paper/`, `blog/`, `results/`, `gates/`)
- the commands in `REPRODUCE2.md` reproduce the theory checks and one figure

Fix what they find; a proof the referee breaks is repaired or its statement is
downgraded. Then write `study2/gates/GATE_4.md` and `FINAL2.md` (summary, the status of
each statement, the outcome of each prediction, issues fixed, known gaps), write
`complete` to `study2/gates/P4.status`, and end the run.
