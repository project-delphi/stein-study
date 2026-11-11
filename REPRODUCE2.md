# Reproducing study 2 (linear theory of JEPA regularisers and its test)

Exact commands, seeds, environment and runtimes for `paper2/paper.pdf`. All commands run from the
repository root. Study 1's guide is `REPRODUCE.md`; its environment section (hardware, Python 3.12.13,
PyTorch 2.15.0, `lejepa` at commit `c293d291ca87`, Quarto 1.10.20 with typst 0.15.1, `requirements.txt`)
applies unchanged. Study 2 installed no new packages. Data: `data/raw/3dshapes.h5` processed as in
`REPRODUCE.md` §2.

## 1. Theory checks (linear-Gaussian model; no images)

Each script runs gradient descent on the linear model from several random starts, compares it with the
closed form or the numerically computed optimum, and writes `results2/theory_<name>.json` with a pass flag.

| command | checks | runtime (M1 Pro, CPU) |
|:--|:--|--:|
| `.venv/bin/python theory/check_t1.py` | Lemma 1, Theorem 1, Corollary 1 (a–d), trap counts | ~4 min |
| `.venv/bin/python theory/check_t2.py` | population regularisers vs batch estimators (Monte Carlo); thresholds t\* (Table 1 of the paper); VICReg scan | ~5 min |
| `.venv/bin/python theory/check_t3.py` | Heuristic 2 (noise on the predictor input). `pass: false` is expected: batch-scaled KSD does not meet the axis-alignment assumption, which is why T3 is labelled heuristic | ~17 min |
| `.venv/bin/python theory/check_t4.py` | Proposition 3 (rotation invariance), Comon check, orthogonal-fit R² | ~5 min |
| `.venv/bin/python -W ignore -m stein.validate2` | theory meets code: the real training loop (`linear` encoder, linear predictor, `cov` arm, synthetic views) against Theorem 1; rerun of a cached study-1 config | seconds (its runs are in the cache) |

Seeds are fixed inside each script. Statements, proofs and status: `theory/theory.md`. Pre-registered
predictions (frozen in P0): `theory/predictions.md`.

## 2. Experiments

All study-2 runs (1075 held-out runs, seeds 1–5, plus reused study-1 runs) are defined in
`stein/grids.py` (`s2_blocks`, `s2_all`). One command runs whatever is missing from the cache and then
the analysis:

```bash
bash study2/jobs/P2.sh          # stein.sweep s2_all --workers 6, then stein.analyze2 (~8 h from an empty cache)
.venv/bin/python -W ignore -m stein.analyze2    # results2/{runs2,cells2,thresholds,frontier}.parquet, analysis.md, schema.json
.venv/bin/python -W ignore -m stein.verdicts2   # results2/verdicts.json (statistic behind every prediction)
```

The cache is `runs/<cfg_hash>/result.json` (or `$STEIN_RUNS`), so finished runs are skipped. Training is
deterministic on Apple MPS, the device every stored run used (`device: mps` in each `result.json`): a re-run
reproduces the stored metrics exactly (below). Other devices (CPU, CUDA) are untested and may differ in the
last digits.

Independent check (P4 review, `study2/review/EMPIRICAL_REVIEW.md`): `check_t1`–`check_t4` and
`stein.validate2` were rerun and reproduced their JSON outputs except `runtime_s`. All 100 Figure 6 runs were
rerun into an empty `STEIN_RUNS` and matched the stored results in every value (12,400 values, list metrics
included). Rerunning `analyze2`, `verdicts2` and `report2` on them gave identical tables and an identical
`fig6_e4.svg`, apart from SVG ids and the date stamp.

### Re-running one figure from scratch: Figure 6 (E4, noise on the predictor input)

Figure 6 rests on 100 runs (E4 block: KSD λ ∈ {0.1, 0.2} and VICReg λ ∈ {1.25, 5}, σ ∈ {0.5, 1},
batch- and fixed-scale noise; plus the σ = 0 configs at the same λ; 5 seeds each). The grid
`s2_fig_e4` lists exactly these. Re-run them into an empty cache and compare them with the stored runs:

```bash
STEIN_RUNS=/tmp/s2_fresh .venv/bin/python -W ignore -m stein.sweep s2_fig_e4 --workers 6   # ~45 min (27 s/run effective)
.venv/bin/python -m stein.repro_check2 /tmp/s2_fresh       # max |diff| over all 47 scalar metrics per run; exit 0 if <= 1e-6
```

For a quick check, add `--limit 2` to the sweep (about 1 min). In P3 this gave 2 runs with
max |diff| = 0. When the re-run metrics match, the figure follows from the stored runs:

```bash
.venv/bin/python -W ignore -m stein.analyze2 && .venv/bin/python -W ignore -m stein.verdicts2
.venv/bin/python -W ignore -m stein.report2     # writes paper2/figures/fig6_e4.pdf among the others
```

## 3. The paper

```bash
.venv/bin/python -W ignore -m stein.report2     # paper2/_generated/{numbers.yml,tab_*.md}, paper2/figures/fig1-6.{pdf,svg} (~10 s)
.venv/bin/python paper2/build_bib.py            # paper2/refs.bib from the verified notes/*.md (+ the companion study)
quarto render paper2/paper.qmd                  # paper2/paper.pdf (typst, ~5 s)
```

`paper2/paper.qmd` executes no Python. Every number in it is a `{{< meta n.KEY >}}` lookup into
`paper2/_generated/numbers.yml`, written by `stein/report2.py` from `results2/` (and, for the motivating
study-1 numbers, `results/lam_sens_summary.parquet`). `paper2/` is rendered on its own and is not in
`_quarto.yml`, so study 1's documents render as before.

## 4. Tests

```bash
.venv/bin/python -m pytest -q tests     # 40 tests, including the cache-hash test over every runs/*/result.json
```
