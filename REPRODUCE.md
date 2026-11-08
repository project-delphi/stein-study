# Reproducing the Stein/JEPA study

This file gives the exact commands, seeds, environment and hardware for every result in
`paper/paper.pdf` and `blog/index.html`. All commands run from the repository root. (`README.md`
belongs to the unattended runner that drove the study and is not part of the reproduction.)

## 1. Environment

| item | value |
|:--|:--|
| hardware | Apple M1 Pro (10 CPU cores, 8 performance), 16 GB RAM, Apple GPU via PyTorch MPS |
| OS | macOS (Darwin 25.6) |
| Python | 3.12.13 in `.venv/` |
| PyTorch | 2.15.0 (MPS backend) |
| SIGReg reference | `lejepa` from github.com/rbalestr-lab/lejepa at commit `c293d291ca87` |
| other packages | pinned in `requirements.txt` (`uv pip freeze` output) |
| renderer | Quarto 1.10.20 with its bundled typst 0.15.1 (no LaTeX needed) |

Create the environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Every training command below uses the `mps` device and 6 worker processes with one CPU thread each.
That is the setting the P1 pilot chose (about 3× the throughput of 8 CPU processes). On a machine
without MPS, change `device="mps"` to `"cpu"` in `stein/grids.py` (and `device:` in `configs/*.yaml`).
The device is not part of the run hash. Expect different speed and possibly small numerical
differences.

## 2. Data

Place the raw files in `data/raw/` (the study used exactly these):
- `3dshapes.h5` (Burgess & Kim 2018);
- `cifar-10-python.tar.gz`;
- `mnist/*-ubyte.gz` (preprocessed but unused: the temporal tier uses 3DShapes clips).

Preprocess them once into memory-mapped uint8 arrays (idempotent; about 1.6 GB in `data/processed/`):

```bash
.venv/bin/python -m stein.prep
```

## 3. Tests and validation gates

```bash
.venv/bin/python -m pytest -q tests          # 28 tests (one or more per loss, gradchecks, sampler, probes)
.venv/bin/python -m stein.validate           # -> results/validation.json (validation gates, GATE_1)
```

## 4. Full pipeline

Runs are keyed by a hash of their full configuration and cached in `runs/<hash>/result.json`. A
finished run is never retrained, so every sweep command can be interrupted and rerun. Set
`STEIN_RUNS=<dir>` to use a different run cache (for example an empty one, as in section 5).

| step | command | output | runs | measured wall clock (6 workers) |
|:--|:--|:--|--:|--:|
| tuning (seeds 100, 101) | `.venv/bin/python -m stein.sweep tune --workers 6 --out results/tuning.parquet` | `results/tuning.parquet` | 192 | 1.4 h (26 s/run effective) |
| level selection | `.venv/bin/python -m stein.select_tuning` | `results/tuning_selected.json`, `results/tuning_table.md` | — | seconds |
| held-out (seeds 1–5) | `bash jobs/P2.sh` (= `.venv/bin/python -m stein.sweep heldout --workers 6 --out results/heldout.parquet`) | `results/heldout.parquet` | 495 | 3.0 h (22 s/run effective) |
| λ sensitivity (seeds 1–5) | `.venv/bin/python -m stein.sweep lam_sens --workers 6 --out results/lam_sens.parquet` | `results/lam_sens.parquet` | 90 (60 new) | 25 min |
| analysis | `.venv/bin/python -W ignore -m stein.analyze` | `results/analysis.md`, `results/summary.parquet`, `results/contrasts.parquet`, `results/*_summary.parquet` | — | < 1 min |
| figures, tables, numbers | `.venv/bin/python -W ignore -m stein.report` | `paper/figures/`, `paper/_generated/`, `blog/data/`, `blog/figures/` | — | < 1 min |
| paper and blog | `quarto render` | `paper/paper.pdf`, `blog/index.html` | — | about 30 s |

Run times come from `logs/compute-P2-1.log`, `logs/compute-P2-2.log` and `logs/lam_sens.log`.
The P1 pilots (`configs/pilot/*.yaml`, run as `.venv/bin/python -m stein.sweep configs/pilot/<name>.yaml
--workers 6 --out results/pilot_<name>.parquet`) informed the defaults. No reported number comes from
them.

`quarto render` executes no Python. The paper reads its numbers from `paper/_generated/numbers.yml`
(`{{< meta n.KEY >}}` shortcodes) and its tables from `paper/_generated/tab_*.md`. The blog reads
`blog/data/numbers.yml`, `blog/data/*.csv` and `blog/figures/*.svg`, and its interactive charts run in
the browser (Observable JS). To view the blog with working charts, serve the `blog/` folder over HTTP
(for example `quarto preview blog/index.qmd`), because browsers block `FileAttachment` reads from
`file://` URLs.

## 5. Reproduce one figure from scratch (about 16 minutes)

This reproduces the temporal-tier panel of Figure 7 (left) from freshly trained models. It retrains
the 25 held-out runs of hypothesis H5 (5 arms × 5 seeds) in an empty run cache, so nothing is reused
from `runs/`. The tuned levels it uses are in `results/tuning_selected.json`.

```bash
rm -rf state/runs_repro
STEIN_RUNS=state/runs_repro OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m stein.sweep heldout_h5 \
    --workers 6 --out results/repro/h5.parquet
.venv/bin/python -W ignore tools/repro_h5.py results/repro/h5.parquet
```

Outputs:
- `results/repro/fig7_left_repro.pdf`: the reported and reproduced panels side by side;
- `results/repro/h5_compare.md`: per-arm means with 95% bootstrap CIs, reported against reproduced.

On the study machine (P3, 2026-11-08) this took 923 s. The log, `logs/repro_h5.log`, is local only,
because `logs/` is not committed. All 25 runs reproduced
every metric bitwise, and all 15 reproduced arm × metric means equal the reported ones. To regenerate
the paper's own Figure 7 from these runs, rerun `stein.analyze`, `stein.report` and `quarto render`;
the figure is unchanged because the runs are identical.

To go fully from scratch, delete `data/processed/` and `runs/` first and run sections 2–4. The full
pipeline takes about 5 h on the hardware above.

## 6. Seeds and fixed randomness

| what | value |
|:--|:--|
| held-out training seeds (all reported results) | 1, 2, 3, 4, 5 |
| tuning (validation) seeds | 100, 101 |
| pilot seed (P1; never reported) | 0 |
| 3DShapes test split (10% of factor combinations) | `numpy.random.default_rng(12345)` |
| bootstrap in `stein.analyze` (10,000 resamples) | `numpy.random.default_rng(20261108)` |
| Figure 1 sampler (illustration only) | `Shapes(seed=7)` |

Each run seeds `torch` and `numpy` from its config seed. MPS kernels are not guaranteed to be
deterministic across hardware or software versions. On the study machine, reruns were bitwise
identical (section 5).

## 7. Where things are

- `stein/`: package code.
  - `prep.py`: data preprocessing.
  - `data.py`: pair, clip and augmentation samplers.
  - `losses.py`: regularisers and priors.
  - `models.py`: encoder, predictor and decoder.
  - `metrics.py`: probes, identifiability, effective rank, NP-KSD.
  - `train.py`: single-run entry point, `python -m stein.train <cfg.yaml> k=v ...`.
  - `sweep.py`: parallel grid runner.
  - `grids.py`: the tuning, held-out, λ-sensitivity and reproduction grids.
  - `select_tuning.py`: level selection.
  - `analyze.py`: bootstrap CIs and contrasts.
  - `report.py`: figures, tables and in-text numbers.
  - `validate.py`: validation gates.
- `configs/`: the base and pilot YAML configs.
- `results/`: every result file; the schema is in `results/schema.json`.
- `notes/`: verified related-work notes. `notes/related_work.md` has the overlap table, and
  `notes/_work/build_bib.py` regenerates `paper/refs.bib` from the notes.
- `gates/GATE_*.md`: per-phase records of what was done, decided and found.
